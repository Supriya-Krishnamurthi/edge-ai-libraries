import { Handle, Position } from "@xyflow/react";
import { getHandleLeftPosition } from "../utils/graphLayout";
import { usePipelineEditorContext } from "../PipelineEditorContext.ts";

export const GVAGenAINodeWidth = 300;

type GVAGenAINodeProps = {
  data: {
    model?: string;
    device?: string;
    "frame-rate"?: string;
    "chunk-size"?: string;
  };
};

const GVAGenAINode = ({ data }: GVAGenAINodeProps) => {
  const { simpleGraph } = usePipelineEditorContext();
  const modelValue = data.model || "";

  return (
    <div className="p-4 rounded shadow-md bg-background border border-l-4 border-l-teal-400 min-w-[300px]">
      <div className="flex gap-3">
        <div className="shrink-0 w-10 h-10 rounded bg-teal-100 dark:bg-teal-900 flex items-center justify-center self-center">
          <svg
            className="w-6 h-6 text-teal-600 dark:text-teal-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.75 3v2.25M14.25 3v2.25M9.75 18.75V21M14.25 18.75V21M3 9.75h2.25M3 14.25h2.25M18.75 9.75H21M18.75 14.25H21M7.5 7.5h9v9h-9z"
            />
          </svg>
        </div>

        <div className="flex-1 flex flex-col">
          <div className="text-xl font-bold text-teal-700 dark:text-teal-300">
            {simpleGraph ? "Video Summarization" : "GVAGenAI"}
          </div>

          <div className="flex items-center gap-1 flex-wrap text-xs text-gray-700 dark:text-gray-300">
            {data.device && <span>{data.device}</span>}

            {modelValue && (
              <>
                {data.device && <span className="text-gray-400">•</span>}
                <span className="truncate max-w-[180px]" title={modelValue}>
                  {modelValue.split("/").pop() || modelValue}
                </span>
              </>
            )}

            {data["frame-rate"] && (
              <>
                {(data.device || modelValue) && (
                  <span className="text-gray-400">•</span>
                )}
                <span>fps: {data["frame-rate"]}</span>
              </>
            )}

            {data["chunk-size"] && (
              <>
                {(data.device || modelValue || data["frame-rate"]) && (
                  <span className="text-gray-400">•</span>
                )}
                <span>chunk: {data["chunk-size"]}</span>
              </>
            )}
          </div>
        </div>
      </div>

      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 bg-teal-500!"
        style={{ left: getHandleLeftPosition("gvagenai") }}
      />

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 bg-teal-500!"
        style={{ left: getHandleLeftPosition("gvagenai") }}
      />
    </div>
  );
};

export default GVAGenAINode;
