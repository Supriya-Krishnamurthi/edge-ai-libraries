import io
import zipfile

import httpx
import pytest

from src.plugins.geti_plugin import GetiPlugin


def model_archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("models/vehicle.xml", "xml")
        archive.writestr("models/vehicle.bin", "bin")
    return stream.getvalue()


@pytest.fixture
def rest_client(monkeypatch):
    requests = []
    models = [{
        "id": "model-1", "name": "Vehicle Detector", "task": "detection",
        "variants": [
            {"id": "variant-fp16", "model_format": "OpenVINO", "precision": ["FP16"]},
            {"id": "variant-int8", "model_format": "OpenVINO", "precision": ["INT8"]},
        ],
    }]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/projects":
            return httpx.Response(200, json=[{"id": "project-1", "name": "Vision"}])
        if request.url.path == "/api/projects/project-1/models":
            return httpx.Response(200, json=models)
        if request.url.path == "/api/projects/project-1/models/model-1":
            return httpx.Response(200, json=models[0])
        if request.url.path.endswith("/variants/variant-fp16/binary"):
            return httpx.Response(200, content=model_archive())
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://geti.test/api")

    async def get_client(session):
        session.http_client = client
        return client

    monkeypatch.setattr(GetiPlugin, "_get_http_client", get_client)
    return requests


@pytest.mark.asyncio
async def test_list_models_uses_project_model_api(rest_client):
    plugin = GetiPlugin()
    session = plugin._build_session({"GETI_HOST": "https://geti.test"})
    result = await plugin._list_models_async(session, {"precision": "FP16"})

    assert result["total"] == 1
    assert result["items"][0]["name"] == "Vehicle Detector"
    assert result["items"][0]["model_type"] == "detection"
    assert result["items"][0]["metadata"]["variant_ids"] == ["variant-fp16"]
    assert [str(request.url) for request in rest_client] == [
        "https://geti.test/api/projects",
        "https://geti.test/api/projects/project-1/models",
    ]


@pytest.mark.asyncio
async def test_download_uses_model_variant_binary_and_extracts(rest_client, tmp_path):
    plugin = GetiPlugin()
    session = plugin._build_session({"GETI_HOST": "https://geti.test"})
    download_dir = []
    path, error, ignored = await plugin.download_model_from_geti(
        session, "model-1", str(tmp_path), "Vehicle Detector",
        project_id="project-1", precision="FP16", model_format="OpenVINO",
        _model_download_dir=download_dir,
    )

    assert error is None
    assert ignored == []
    assert path == str(tmp_path / "geti" / "vehicle detector" / "fp16")
    assert (tmp_path / "geti" / "vehicle detector" / "fp16" / "vehicle.xml").read_text() == "xml"
    assert download_dir == [path]
    assert str(rest_client[-1].url) == (
        "https://geti.test/api/projects/project-1/models/model-1/variants/variant-fp16/binary"
    )


def test_geti_config_has_no_workspace_and_token_is_optional():
    keys = {key.name: key for key in GetiPlugin().hub_config_keys()}
    assert set(keys) == {"GETI_HOST", "GETI_TOKEN"}
    assert keys["GETI_HOST"].required is True
    assert keys["GETI_TOKEN"].required is False