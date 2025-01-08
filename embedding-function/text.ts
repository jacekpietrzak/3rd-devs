import { createByModelName } from "@microsoft/tiktokenizer";
import type { Document, Headers } from "./types";

const SPECIAL_TOKENS = new Map<string, number>([
  ["<|im_start|>", 100264],
  ["<|im_end|>", 100265],
  ["<|im_sep|>", 100266],
]);

export const createTokenizer = async (modelName: string = "gpt-4") => {
  return await createByModelName(modelName, SPECIAL_TOKENS);
};

const formatForTokenization = (text: string): string => {
  return `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant<|im_end|>`;
};

export const countTokens = (
  tokenizer: Awaited<ReturnType<typeof createByModelName>>,
  text: string
): number => {
  const formattedContent = formatForTokenization(text);
  const tokens = tokenizer.encode(
    formattedContent,
    Array.from(SPECIAL_TOKENS.keys())
  );
  return tokens.length;
};

export const extractHeaders = (text: string): Headers => {
  const headers: Headers = {};
  const headerRegex = /(^|\n)(#{1,6})\s+(.*)/g;
  let match;

  while ((match = headerRegex.exec(text)) !== null) {
    const level = match[2].length;
    const content = match[3].trim();
    const key = `h${level}`;
    headers[key] = headers[key] || [];
    headers[key].push(content);
  }

  return headers;
};

export const extractUrlsAndImages = (
  text: string
): { content: string; urls: string[]; images: string[] } => {
  const urls: string[] = [];
  const images: string[] = [];
  let urlIndex = 0;
  let imageIndex = 0;

  const content = text
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_match, altText, url) => {
      images.push(url);
      return `![${altText}]({{$img${imageIndex++}}})`;
    })
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, linkText, url) => {
      urls.push(url);
      return `[${linkText}]({{$url${urlIndex++}}})`;
    });

  return { content, urls, images };
};

export const createDocument = async (
  text: string,
  options: {
    model?: string;
    additionalMetadata?: Record<string, any>;
  } = {}
): Promise<Document> => {
  const tokenizer = await createTokenizer(options.model);
  const tokens = countTokens(tokenizer, text);
  const headers = extractHeaders(text);
  const { content, urls, images } = extractUrlsAndImages(text);

  return {
    text: content,
    metadata: {
      tokens,
      headers,
      urls,
      images,
      ...options.additionalMetadata,
    },
  };
};
