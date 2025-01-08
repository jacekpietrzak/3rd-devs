import { createOpenAIClient } from "./openai";
import { createDocument } from "./text";
import {
  createVectorClient,
  ensureCollection,
  addPoints,
  performSearch,
} from "./vector";
import type { Config } from "./types";

const data = [
  "Apple (Consumer Electronics)",
  "Tesla (Automotive)",
  "Microsoft (Software)",
  "Google (Internet Services)",
  "Nvidia (Semiconductors)",
  "Meta (Social Media)",
  "X Corp (Social Media)",
  "Tech•sistence (Newsletter)",
];

const queries = ["Car company", "Macbooks", "Facebook", "Newsletter"];

const COLLECTION_NAME = "aidevs";

const config: Config = {
  openAIApiKey: process.env.OPENAI_API_KEY,
  qdrantUrl: process.env.QDRANT_URL,
  qdrantApiKey: process.env.QDRANT_API_KEY,
  jinaApiKey: process.env.JINA_API_KEY,
};

const initializeData = async () => {
  const openAIClient = createOpenAIClient(config);
  const vectorClient = createVectorClient(config);

  const points = await Promise.all(
    data.map(async (text) => {
      return await createDocument(text, {
        model: "gpt-4",
        additionalMetadata: { role: "embedding-test" },
      });
    })
  );

  await ensureCollection(vectorClient, COLLECTION_NAME);
  await addPoints(vectorClient, openAIClient, COLLECTION_NAME, points);
};

const main = async () => {
  const openAIClient = createOpenAIClient(config);
  const vectorClient = createVectorClient(config);

  await initializeData();

  const searchResults = await Promise.all(
    queries.map((query) =>
      performSearch(vectorClient, openAIClient, COLLECTION_NAME, query, 3)
    )
  );

  queries.forEach((query, index) => {
    console.log(`Query: ${query}`);
    searchResults[index].forEach((result, resultIndex) => {
      console.log(
        `  ${resultIndex + 1}. ${result.payload.text} (Score: ${result.score})`
      );
    });
    console.log();
  });
};

main().catch(console.error);
