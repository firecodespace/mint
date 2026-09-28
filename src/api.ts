// Thin wrappers over Tauri commands exposed by the Rust core.
import { invoke } from "@tauri-apps/api/core";
import type {
  Memory,
  NewMemory,
  SearchRequest,
  SearchResponse,
  Stats,
} from "./types";

export function addMemory(input: NewMemory): Promise<Memory> {
  return invoke<Memory>("add_memory", { input });
}

export function searchMemories(req: SearchRequest): Promise<SearchResponse> {
  return invoke<SearchResponse>("search_memories", { req });
}

export function listMemories(): Promise<Memory[]> {
  return invoke<Memory[]>("list_memories");
}

export function deleteMemory(id: string): Promise<void> {
  return invoke<void>("delete_memory", { id });
}

export function getStats(): Promise<Stats> {
  return invoke<Stats>("stats");
}
