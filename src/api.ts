// Thin wrappers over Tauri commands exposed by the Rust core.
import { invoke } from "@tauri-apps/api/core";
import type {
  ChatMessage,
  ChatStatus,
  ChatTurnResult,
  Conversation,
  ConversationSummary,
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

export function chatStatus(): Promise<ChatStatus> {
  return invoke<ChatStatus>("chat_status");
}

/** Run one chat turn. Streaming deltas arrive via Tauri events
 * (chat:stage, chat:token, chat:captured); this resolves with the final turn. */
export function chat(
  conversationId: string,
  message: string,
  history: ChatMessage[],
): Promise<ChatTurnResult> {
  return invoke<ChatTurnResult>("chat", { conversationId, message, history });
}

// ---- conversations -------------------------------------------------------

export function listConversations(): Promise<ConversationSummary[]> {
  return invoke<ConversationSummary[]>("list_conversations");
}

export function getConversation(id: string): Promise<Conversation | null> {
  return invoke<Conversation | null>("get_conversation", { id });
}

export function createConversation(): Promise<Conversation> {
  return invoke<Conversation>("create_conversation");
}

export function renameConversation(id: string, title: string): Promise<void> {
  return invoke<void>("rename_conversation", { id, title });
}

export function deleteConversation(id: string, deleteMemories: boolean): Promise<void> {
  return invoke<void>("delete_conversation", { id, deleteMemories });
}
