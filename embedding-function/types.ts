export interface Config {
  openAIApiKey?: string;
  qdrantUrl?: string;
  qdrantApiKey?: string;
  jinaApiKey?: string;
}

export interface Document {
  text: string;
  metadata: {
    tokens: number;
    headers: Headers;
    urls: string[];
    images: string[];
    [key: string]: any;
  };
}

export interface Headers {
  [key: string]: string[];
}

export interface VectorPoint {
  id?: string;
  text: string;
  metadata?: Record<string, any>;
}

export interface SearchResult {
  id: string | number;
  score: number;
  payload: {
    text: string;
    [key: string]: any;
  };
  vector?: number[];
}
