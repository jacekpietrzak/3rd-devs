import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { CreateEmbeddingResponse } from "openai/resources/embeddings";
import type { Config } from "./types";

export const createOpenAIClient = (config: Config) => {
  return new OpenAI({ apiKey: config.openAIApiKey });
};

export const createEmbedding = async (
  client: OpenAI,
  text: string
): Promise<number[]> => {
  try {
    const response: CreateEmbeddingResponse = await client.embeddings.create({
      model: "text-embedding-3-large",
      input: text,
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error("Error creating embedding:", error);
    throw error;
  }
};

export const createJinaEmbedding = async (
  text: string,
  config: Config
): Promise<number[]> => {
  try {
    const response = await fetch("https://api.jina.ai/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.jinaApiKey}`,
      },
      body: JSON.stringify({
        model: "jina-embeddings-v3",
        task: "text-matching",
        dimensions: 1024,
        late_chunking: false,
        embedding_type: "float",
        input: [text],
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
  } catch (error) {
    console.error("Error creating Jina embedding:", error);
    throw error;
  }
};

export const completion = async (
  client: OpenAI,
  messages: ChatCompletionMessageParam[],
  options: {
    model?: string;
    stream?: boolean;
    jsonMode?: boolean;
    maxTokens?: number;
  } = {}
) => {
  const {
    model = "gpt-4o-mini",
    stream = false,
    jsonMode = false,
    maxTokens = 4096,
  } = options;

  try {
    const chatCompletion = await client.chat.completions.create({
      messages,
      model,
      ...(model !== "o1-mini" &&
        model !== "o1-preview" && {
          stream,
          max_tokens: maxTokens,
          response_format: jsonMode
            ? { type: "json_object" }
            : { type: "text" },
        }),
    });

    return chatCompletion;
  } catch (error) {
    console.error("Error in OpenAI completion:", error);
    throw error;
  }
};

export const calculateImageTokens = (
  width: number,
  height: number,
  detail: "low" | "high"
): number => {
  if (detail === "low") return 85;

  const MAX_DIMENSION = 2048;
  const SCALE_SIZE = 768;

  // Resize to fit within MAX_DIMENSION x MAX_DIMENSION
  if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
    const aspectRatio = width / height;
    if (aspectRatio > 1) {
      width = MAX_DIMENSION;
      height = Math.round(MAX_DIMENSION / aspectRatio);
    } else {
      height = MAX_DIMENSION;
      width = Math.round(MAX_DIMENSION * aspectRatio);
    }
  }

  // Scale the shortest side to SCALE_SIZE
  if (width >= height && height > SCALE_SIZE) {
    width = Math.round((SCALE_SIZE / height) * width);
    height = SCALE_SIZE;
  } else if (height > width && width > SCALE_SIZE) {
    height = Math.round((SCALE_SIZE / width) * height);
    width = SCALE_SIZE;
  }

  // Calculate the number of 512px squares
  const numSquares = Math.ceil(width / 512) * Math.ceil(height / 512);

  // Calculate the token cost
  return numSquares * 170 + 85;
};
