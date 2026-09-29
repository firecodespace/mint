// Thin wrappers over Tauri commands exposed by the Rust core.
import { invoke } from "@tauri-apps/api/core";
import type {
  ChatMessage,
  ChatStatus,
  ChatTurnResult,
  Conversation,
  ConversationSummary,
  DocIngestResult,
  GraphData,
  MaintenanceReport,
  Memory,
  NewMemory,
  OrganizeReport,
  PolicySummary,
  SearchRequest,
  SearchResponse,
  Stats,
  SyncDecision,
  SyncReport,
  SyncStatus,
  TopicInfo,
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

// ---- graph & vault -------------------------------------------------------

export function graphData(): Promise<GraphData> {
  return invoke<GraphData>("graph_data");
}

export function ingestDocument(name: string, dataBase64: string): Promise<DocIngestResult> {
  return invoke<DocIngestResult>("ingest_document", { name, dataBase64 });
}

export function listDocuments(): Promise<Memory[]> {
  return invoke<Memory[]>("list_documents");
}

export function deleteDocument(id: string): Promise<void> {
  return invoke<void>("delete_document", { id });
}

// ---- sync ----------------------------------------------------------------

export function syncStatus(): Promise<SyncStatus> {
  return invoke<SyncStatus>("sync_status");
}

export function setOnline(online: boolean): Promise<void> {
  return invoke<void>("set_online", { online });
}

export function setServerUrl(url: string): Promise<void> {
  return invoke<void>("set_server_url", { url });
}

export function syncNow(): Promise<SyncReport> {
  return invoke<SyncReport>("sync_now");
}

export function setAutoSync(enabled: boolean): Promise<void> {
  return invoke<void>("set_auto_sync", { enabled });
}

export function setPullAll(enabled: boolean): Promise<void> {
  return invoke<void>("set_pull_all", { enabled });
}

/** What stays on this device and why (sync policy overview). */
export function policySummary(): Promise<PolicySummary> {
  return invoke<PolicySummary>("policy_summary");
}

/** Allow a memory to sync, or keep it on this device. */
export function setSyncOverride(id: string, share: boolean): Promise<SyncDecision> {
  return invoke<SyncDecision>("set_sync_override", { id, share });
}

/** A memory's version chain, oldest first. */
export function versionChain(id: string): Promise<Memory[]> {
  return invoke<Memory[]>("version_chain", { id });
}

// ---- maintenance ---------------------------------------------------------

export function runMaintenance(): Promise<MaintenanceReport> {
  return invoke<MaintenanceReport>("run_maintenance");
}

export function clearArchive(): Promise<number> {
  return invoke<number>("clear_archive");
}

// ---- topics (schema layer) -----------------------------------------------

export function listTopics(): Promise<TopicInfo[]> {
  return invoke<TopicInfo[]>("list_topics");
}

export function renameTopic(id: string, name: string): Promise<void> {
  return invoke<void>("rename_topic", { id, name });
}

export function mergeTopics(from: string, into: string): Promise<void> {
  return invoke<void>("merge_topics", { from, into });
}

export function moveToTopic(memoryId: string, topicId: string): Promise<void> {
  return invoke<void>("move_to_topic", { memoryId, topicId });
}

export function refreshTopic(id: string): Promise<TopicInfo> {
  return invoke<TopicInfo>("refresh_topic", { id });
}

/** Route every not-yet-organized memory/document into topics, then summarize. */
export function organizeTopics(): Promise<OrganizeReport> {
  return invoke<OrganizeReport>("organize_topics");
}

// ---- timeline ------------------------------------------------------------

export function listScheduled(): Promise<Memory[]> {
  return invoke<Memory[]>("list_scheduled");
}

export function setTaskDone(id: string, done: boolean): Promise<void> {
  return invoke<void>("set_task_done", { id, done });
}
