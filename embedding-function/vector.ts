import { QdrantClient } from "@qdrant/js-client-rest";
import { v4 as uuidv4 } from "uuid";
import fs from "fs/promises";
import path from "path";
import type { Config, VectorPoint, SearchResult } from "./types";
import { createEmbedding } from "./openai";
import OpenAI from "openai";

export const createVectorClient = (config: Config) => {
  return new QdrantClient({
    url: config.qdrantUrl,
    apiKey: config.qdrantApiKey,
  });
};

export const ensureCollection = async (client: QdrantClient, name: string) => {
  const collections = await client.getCollections();
  if (!collections.collections.some((c) => c.name === name)) {
    await client.createCollection(name, {
      vectors: { size: 3072, distance: "Cosine" },
    });
  }
};

export const addPoints = async (
  client: QdrantClient,
  openAIClient: OpenAI,
  collectionName: string,
  points: VectorPoint[]
) => {
  const pointsToUpsert = await Promise.all(
    points.map(async (point) => {
      const embedding = await createEmbedding(openAIClient, point.text);

      return {
        id: point.id || uuidv4(),
        vector: embedding,
        payload: {
          text: point.text,
          ...point.metadata,
        },
      };
    })
  );

  const pointsFilePath = path.join(process.cwd(), "points.json");
  await fs.writeFile(pointsFilePath, JSON.stringify(pointsToUpsert, null, 2));

  await client.upsert(collectionName, {
    wait: true,
    points: pointsToUpsert,
  });
};

export const performSearch = async (
  client: QdrantClient,
  openAIClient: OpenAI,
  collectionName: string,
  query: string,
  limit: number = 5
): Promise<SearchResult[]> => {
  const queryEmbedding = await createEmbedding(openAIClient, query);
  const results = await client.search(collectionName, {
    vector: queryEmbedding,
    limit,
    with_payload: true,
  });

  return results.map((result) => ({
    id: String(result.id),
    payload: {
      text: String(result.payload?.text || ""),
      ...result.payload,
    },
    score: result.score,
    vector:
      Array.isArray(result.vector) &&
      result.vector.every((x) => typeof x === "number")
        ? result.vector
        : undefined,
  }));
};
