import { DEVICE_TYPES } from "@/features/pipeline-editor/nodes/shared-types.ts";

export const gvaGenAIConfig = {
  editableProperties: [
    {
      key: "model",
      label: "Model",
      type: "select" as const,
      defaultValue: "",
      description: "OpenVINO GenAI model",
      params: {
        filter: "genai",
      },
    },
    {
      key: "device",
      label: "Device",
      type: "select" as const,
      options: DEVICE_TYPES,
      description: "Target device for inference",
    },
    {
      key: "frame-rate",
      label: "Inference interval",
      type: "text" as const,
      defaultValue: "1",
      description: "Frame sampling rate (fps) used by gvagenai",
    },
    {
      key: "chunk-size",
      label: "Batch size",
      type: "text" as const,
      defaultValue: "4",
      description: "Number of sampled frames per inference call",
    },
  ],
};
