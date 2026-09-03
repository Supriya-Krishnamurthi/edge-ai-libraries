# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shutil
import tarfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.core.interfaces import DownloadTask, ModelDownloadPlugin, PluginConfigKey
from src.utils.logging import logger

DEFAULT_MODEL_FORMAT = "OpenVINO"
DEFAULT_PRECISION = "FP16"
DEFAULT_EXPORT_TYPE = "optimized"
GETI_LISTING_FILTER_FIELDS = [
    "project_id", "project_name", "model_name", "export_type", "precision",
    "model_format", "architecture", "variant_id",
]


class _GetiSession:
    """Per-request REST configuration and lazily-created HTTP client."""

    def __init__(self, server_url: Optional[str], api_token: Optional[str], verify_ssl: bool) -> None:
        self.server_url = server_url.rstrip("/") if server_url else None
        self.api_token = api_token
        self.verify_ssl = verify_ssl
        self.http_client: Optional[httpx.AsyncClient] = None


class GetiPlugin(ModelDownloadPlugin):
    """Download and list models through the Geti 3.0 Model REST API."""

    _instance: Optional["GetiPlugin"] = None

    def __new__(cls) -> "GetiPlugin":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._req_timeout = 30.0

    def hub_config_keys(self, hub: str = "geti") -> List[PluginConfigKey]:
        return [
            PluginConfigKey("GETI_HOST", "Geti server URL.", sensitive=True, required=True, group="geti"),
            PluginConfigKey("GETI_TOKEN", "Geti personal access token.", sensitive=True, group="geti"),
        ]

    @staticmethod
    def _parse_bool(value: Any, ignore_empty: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        value = value or ""
        if ignore_empty and not value:
            return True
        return str(value).lower() in ("true", "1", "yes", "on")

    def _build_session(self, resolved_config: Dict[str, Any]) -> _GetiSession:
        return _GetiSession(
            resolved_config.get("GETI_HOST"), resolved_config.get("GETI_TOKEN"),
            self._parse_bool(os.getenv("GETI_SERVER_SSL_VERIFY", "False"), ignore_empty=True),
        )

    def _session_from_kwargs(self, kwargs: Dict[str, Any]) -> _GetiSession:
        resolved = kwargs.get("resolved_config")
        if resolved is None:
            resolved = self.resolve_config(kwargs.get("override_credentials"))
        return self._build_session(resolved)

    async def _get_http_client(self, session: _GetiSession) -> httpx.AsyncClient:
        if session.http_client is None:
            if not session.server_url:
                raise ValueError("GETI_HOST is required")
            headers = {"Authorization": f"Bearer {session.api_token}"} if session.api_token else {}
            session.http_client = httpx.AsyncClient(
                base_url=f"{session.server_url}/api", headers=headers,
                verify=session.verify_ssl, timeout=self._req_timeout,
            )
        return session.http_client

    async def _request(self, session: _GetiSession, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = await (await self._get_http_client(session)).request(method, path, **kwargs)
        response.raise_for_status()
        return response

    async def _close_session(self, session: _GetiSession) -> None:
        if session.http_client is not None:
            await session.http_client.aclose()
            session.http_client = None

    @property
    def plugin_name(self) -> str:
        return "geti"

    @property
    def plugin_type(self) -> str:
        return "downloader"

    @property
    def supports_listing(self) -> bool:
        return True

    @property
    def listing_filter_fields(self) -> List[str]:
        return GETI_LISTING_FILTER_FIELDS

    def can_handle(self, model_name: str, hub: str, **kwargs: Any) -> bool:
        return hub.lower() == "geti"

    def validate_credentials(self, resolved_config: Dict[str, Any], timeout: int = 5) -> Dict[str, Any]:
        host = resolved_config.get("GETI_HOST")
        if not host:
            return {"name": "geti_config", "ok": False, "message": "GETI_HOST is required."}
        session = self._build_session(resolved_config)

        async def check() -> Dict[str, Any]:
            headers = {"Authorization": f"Bearer {session.api_token}"} if session.api_token else {}
            try:
                async with httpx.AsyncClient(base_url=session.server_url, headers=headers,
                                              verify=session.verify_ssl, timeout=timeout) as client:
                    response = await client.get("/health")
                    response.raise_for_status()
                return {"name": "geti_auth", "ok": True, "message": f"Connected to {host}."}
            except httpx.HTTPStatusError as exc:
                return {"name": "geti_auth", "ok": False, "message": f"Geti API error: {exc.response.status_code}"}
            except Exception as exc:
                return {"name": "geti_connectivity", "ok": False, "message": f"Cannot connect to Geti at {host}: {exc}"}

        return asyncio.run(check())

    async def get_projects(self, session: _GetiSession, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        payload = (await self._request(session, "GET", "/projects")).json()
        projects = payload.get("items", payload) if isinstance(payload, dict) else payload
        return [{
            "id": project.get("id"), "name": project.get("name", project.get("id")),
            "creation_time": project.get("creation_time"), "project": project,
        } for project in projects if project_id is None or project.get("id") == project_id]

    async def _get_project_models(self, session: _GetiSession, project_id: str) -> List[Dict[str, Any]]:
        payload = (await self._request(session, "GET", f"/projects/{project_id}/models")).json()
        return payload.get("items", payload) if isinstance(payload, dict) else payload

    @staticmethod
    def _variants(model: Dict[str, Any]) -> List[Dict[str, Any]]:
        variants = model.get("variants", model.get("model_variants", []))
        return variants if isinstance(variants, list) else []

    @staticmethod
    def _precisions(variant: Dict[str, Any]) -> List[str]:
        value = variant.get("precision")
        values = value if isinstance(value, list) else [value]
        return [str(item) for item in values if item is not None]

    def _filter_variants(self, variants: List[Dict[str, Any]], model_format: Optional[str],
                         precision: Optional[str], extra_filters: Optional[Dict[str, Any]] = None
                         ) -> Tuple[List[Dict[str, Any]], List[str]]:
        filtered = [variant for variant in variants if
                    (not model_format or str(variant.get("model_format", variant.get("format", ""))).lower() == model_format.lower())
                    and (not precision or any(item.lower() == precision.lower() for item in self._precisions(variant)))]
        ignored: List[str] = []
        for key, value in (extra_filters or {}).items():
            if not any(key in variant for variant in filtered):
                ignored.append(key)
                logger.warning(f"Filter field '{key}' is not present in model data. This filter will be ignored.")
                continue
            filtered = [variant for variant in filtered if self._match_filter_value(variant.get(key), value)]
        return filtered, ignored

    @staticmethod
    def _match_filter_value(actual: Any, expected: Any) -> bool:
        if isinstance(expected, str):
            if isinstance(actual, str):
                return actual.lower() == expected.lower()
            if isinstance(actual, list):
                return any(str(item).lower() == expected.lower() for item in actual)
        if isinstance(actual, list):
            return expected in actual
        return actual == expected

    async def _list_models_async(self, session: _GetiSession, filters: Optional[Dict[str, Any]] = None,
                                 limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        filters = filters or {}
        self._validate_listing_filters(filters)
        projects = await self.get_projects(session, filters.get("project_id"))
        project_name = str(filters["project_name"]).lower() if filters.get("project_name") is not None else None
        model_name = str(filters["model_name"]).lower() if filters.get("model_name") is not None else None
        model_format, precision = filters.get("model_format"), filters.get("precision")
        export_type = str(filters.get("export_type") or DEFAULT_EXPORT_TYPE).lower()
        items = []
        for project in projects:
            if project_name and project_name not in str(project["name"]).lower():
                continue
            for model in await self._get_project_models(session, project["id"]):
                if model_name and model_name not in str(model.get("name", "")).lower():
                    continue
                variants = self._variants(model)
                if export_type == "optimized":
                    variants, _ = self._filter_variants(variants, model_format, precision)
                    if not variants:
                        continue
                task = model.get("task", model.get("task_type", project["project"].get("task_type")))
                items.append({
                    "name": model.get("name") or model.get("id", ""), "owner": project["name"],
                    "precisions": sorted({item for variant in variants for item in self._precisions(variant)}),
                    "model_type": str(task) if task is not None else None,
                    "last_modified": model.get("last_updated", model.get("updated_at", model.get("creation_time"))),
                    "metadata": {"project_id": project["id"], "project_name": project["name"],
                                 "model_id": model.get("id"),
                                 "variant_ids": [variant.get("id", variant.get("variant_id")) for variant in variants],
                                 "architecture": model.get("architecture")},
                })
        return {"items": items[offset:offset + limit], "total": len(items)}

    def list_models(self, filters: Optional[Dict[str, Any]] = None, limit: int = 50,
                    offset: int = 0, **kwargs: Any) -> Dict[str, Any]:
        session = self._session_from_kwargs(kwargs)
        try:
            return asyncio.run(self._list_models_async(session, filters, limit, offset))
        finally:
            asyncio.run(self._close_session(session))

    async def search_model(self, session: _GetiSession, model_name: str, export_type: Optional[str] = None,
                           precision: Optional[str] = None, revision: Optional[int] = None,
                           model_format: Optional[str] = None, extra_filters: Optional[Dict[str, Any]] = None
                           ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[List[str]]]:
        for project in await self.get_projects(session):
            for model in await self._get_project_models(session, project["id"]):
                if str(model.get("name", "")).lower() != model_name.lower():
                    continue
                variants, ignored = self._filter_variants(self._variants(model), model_format, precision, extra_filters)
                if export_type != "base" and not variants:
                    continue
                variant_id = variants[0].get("id", variants[0].get("variant_id")) if variants else None
                return project["id"], model.get("id"), variant_id, None, ignored
        return None, None, None, f"Model not found: {model_name}", None

    async def get_model_id_by_name(self, session: _GetiSession, project_id: str,
                                   model_group_id: Optional[str], model_name: str) -> Optional[str]:
        for model in await self._get_project_models(session, project_id):
            if str(model.get("name", "")).lower() == model_name.lower():
                return model.get("id")
        return None

    async def select_optimized_model(self, model: Dict[str, Any], optimized_model_id: Optional[str],
                                     precision: Optional[str], base_model_id: str,
                                     model_format: Optional[str] = None,
                                     extra_filters: Optional[Dict[str, Any]] = None
                                     ) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
        variants = self._variants(model)
        if optimized_model_id:
            return next((item for item in variants if item.get("id", item.get("variant_id")) == optimized_model_id), None), None
        if precision or model_format or extra_filters:
            variants, ignored = self._filter_variants(variants, model_format, precision, extra_filters)
            return (variants[0] if variants else None), ignored
        return (variants[0] if variants else None), None

    async def download_model_from_geti(self, session: _GetiSession, model_id: str, output_dir: str,
                                       model_name: str = "", **kwargs: Any) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
        try:
            project_id = kwargs.get("project_id")
            if not project_id:
                return None, "Project ID is required for Geti 3.0 model download", None
            detail = (await self._request(session, "GET", f"/projects/{project_id}/models/{model_id}")).json()
            variant, ignored = await self.select_optimized_model(
                detail, kwargs.get("optimized_model_id"), kwargs.get("precision"), model_id,
                kwargs.get("model_format"), kwargs.get("extra_filters"),
            )
            if variant is None:
                return None, "No model variant matched the requested criteria", ignored
            variant_id = variant.get("id", variant.get("variant_id"))
            response = await self._request(session, "GET", f"/projects/{project_id}/models/{model_id}/variants/{variant_id}/binary")
            model_dir = os.path.join(output_dir, "geti", model_name.lower(),
                                     str(kwargs.get("precision") or DEFAULT_PRECISION).lower())
            os.makedirs(model_dir, exist_ok=True)
            if kwargs.get("_model_download_dir") is not None:
                kwargs["_model_download_dir"].append(model_dir)
            archive_path = os.path.join(model_dir, "model.zip")
            with open(archive_path, "wb") as model_file:
                model_file.write(response.content)
            await self.extract_model_files(model_dir)
            return model_dir, None, ignored
        except httpx.HTTPStatusError as exc:
            return None, f"Geti API error: {exc.response.status_code}", None
        except Exception as exc:
            logger.error(f"Download failed: {type(exc).__name__}: {exc}")
            return None, f"Download failed: {exc}", None

    async def download(self, model_name: str, output_dir: str, **kwargs: Any) -> Dict[str, Any]:
        session = self._session_from_kwargs(kwargs)
        try:
            config = kwargs.get("config", {}) or {}
            supported = {"export_type", "precision", "model_format", "optimized_model_id", "model_id", "project_id"}
            extra_filters = {key: value for key, value in config.items() if key not in supported and value is not None}
            export_type = str(config.get("export_type") or DEFAULT_EXPORT_TYPE).lower()
            precision = str(config.get("precision") or DEFAULT_PRECISION).lower()
            model_format = config.get("model_format") or DEFAULT_MODEL_FORMAT
            model_id, project_id, variant_id, error, ignored = await self.search_model(
                session, model_name, export_type, precision, model_format=model_format, extra_filters=extra_filters
            ) if not config.get("model_id") or not config.get("project_id") else (
                config["model_id"], config["project_id"], config.get("optimized_model_id"), None, None
            )
            if not model_id or not project_id:
                return {"success": False, "error": error or f"Model not found: {model_name}"}
            path, error, download_ignored = await self.download_model_from_geti(
                session, model_id, output_dir, model_name, project_id=project_id, precision=precision,
                model_format=model_format, optimized_model_id=config.get("optimized_model_id") or variant_id,
                extra_filters=extra_filters, _model_download_dir=kwargs.get("_model_download_dir"),
            )
            if not path:
                return {"success": False, "error": error or "Download failed"}
            ignored_fields = sorted(set((ignored or []) + (download_ignored or [])))
            result = {"model_name": model_name, "source": "geti", "download_path": os.path.join(output_dir, "geti"), "success": True}
            if ignored_fields:
                result["warnings"] = {"ignored_fields": ignored_fields}
            return result
        except Exception as exc:
            logger.error(f"Download error: {type(exc).__name__}: {exc}")
            return {"success": False, "error": str(exc)}
        finally:
            await self._close_session(session)

    @staticmethod
    def _safe_extract_archive(archive_path: str, extract_dir: str) -> None:
        root = os.path.realpath(extract_dir)
        safe = lambda name: os.path.realpath(os.path.join(root, name)).startswith(root + os.sep)
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    if safe(member):
                        archive.extract(member, extract_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extract_dir, members=[item for item in archive.getmembers() if safe(item.name)])

    async def extract_model_files(self, model_dir: str) -> None:
        archive_path = os.path.join(model_dir, "model.zip")
        models_dir = os.path.join(model_dir, "models")
        if not os.path.exists(models_dir):
            if os.path.exists(archive_path):
                self._safe_extract_archive(archive_path, model_dir)
                os.remove(archive_path)
            return
        try:
            for item in os.listdir(models_dir):
                source, destination = os.path.join(models_dir, item), os.path.join(model_dir, item)
                if os.path.isdir(source):
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
            shutil.rmtree(models_dir)
        except Exception as exc:
            logger.warning(f"File extraction issue: {exc}")

    def get_download_tasks(self, model_name: str, **kwargs: Any) -> List[DownloadTask]:
        raise NotImplementedError("Geti plugin does not support task-based downloading")

    def download_task(self, task: DownloadTask, output_dir: str, **kwargs: Any) -> str:
        raise NotImplementedError("Geti plugin does not support task-based downloading")

    async def post_process(self, model_name: str, output_dir: str, downloaded_paths: List[str], **kwargs: Any) -> Dict[str, Any]:
        return {"model_name": model_name, "source": "geti", "download_path": output_dir, "success": True}