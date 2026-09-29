import { useEffect, useMemo, useRef, useState } from "react";
import {
  graphData,
  listTopics,
  mergeTopics,
  moveToTopic,
  organizeTopics,
  refreshTopic,
  renameTopic,
} from "../api";
import type { GraphData, Memory, TopicInfo } from "../types";

interface Node {
  id: string;
  kind: string;
  label: string;
  isChunk: boolean;
  salience: number;
  topicId: string | null;
  x: number;
  y: number;
  vx: number;
  vy: number;
  pinned?: boolean;
}
interface Edge {
  a: string;
  b: string;
  relation: string;
}

const W = 1000;
const H = 680;
const REPULSE = 3600;
const SPRING = 0.04;
const LINK_PART = 46;
const LINK_MENTION = 78;
const LINK_REL = 120;
const CENTER = 0.008;
const DAMP = 0.82;
const SETTLE_ITERS = 140;

function linkLength(relation: string) {
  if (relation === "part_of") return LINK_PART;
  if (relation === "mentions") return LINK_MENTION;
  return LINK_REL;
}

function radius(n: Node, selected: boolean) {
  if (n.isChunk) return selected ? 6 : 3.5;
  const base =
    n.kind === "topic"
      ? 11
      : n.kind === "document"
        ? 9
        : n.kind === "entity" || n.kind === "summary"
          ? 8
          : 6;
  const scaled = base * (0.7 + 0.6 * (n.salience || 0.5));
  return selected ? scaled + 3 : scaled;
}

/** One physics step over the node set. Mutates node positions in place. */
function tick(nodes: Node[], edges: Edge[], byId: Map<string, Node>) {
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      let dx = a.x - b.x;
      let dy = a.y - b.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 0.01) {
        dx = Math.random();
        dy = Math.random();
        d2 = 1;
      }
      const f = REPULSE / d2;
      const d = Math.sqrt(d2);
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }
  }
  for (const e of edges) {
    const a = byId.get(e.a);
    const b = byId.get(e.b);
    if (!a || !b) continue;
    const target = linkLength(e.relation);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const diff = ((dist - target) / dist) * SPRING;
    a.vx += dx * diff;
    a.vy += dy * diff;
    b.vx -= dx * diff;
    b.vy -= dy * diff;
  }
  for (const n of nodes) {
    if (n.pinned) {
      n.vx = 0;
      n.vy = 0;
      continue;
    }
    n.vx += (W / 2 - n.x) * CENTER;
    n.vy += (H / 2 - n.y) * CENTER;
    n.vx *= DAMP;
    n.vy *= DAMP;
    n.x += n.vx;
    n.y += n.vy;
  }
}

export function MemoryGraph({ memories }: { memories: Memory[] }) {
  const [data, setData] = useState<GraphData | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, w: W, h: H });
  const [selected, setSelected] = useState<string | null>(null);
  const dragId = useRef<string | null>(null);
  const pan = useRef<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const gRefs = useRef<Map<string, SVGGElement | null>>(new Map());
  const lineRefs = useRef<(SVGLineElement | null)[]>([]);
  const rafRef = useRef(0);
  const coolRef = useRef(0);

  // Topics (schema layer): legend list, focus highlight, and override actions.
  const [topics, setTopics] = useState<TopicInfo[]>([]);
  const [focusTopic, setFocusTopic] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState("");
  const [mergeTarget, setMergeTarget] = useState("");

  useEffect(() => {
    graphData().then(setData).catch(console.error);
    listTopics()
      .then(setTopics)
      .catch(() => setTopics([]));
  }, [memories.length, reload]);

  useEffect(() => {
    setRenameVal("");
    setMergeTarget("");
    setActionError(null);
  }, [selected]);

  /** Run a topic action, then refetch graph + topics. */
  async function act(kind: string, fn: () => Promise<unknown>) {
    setBusy(kind);
    setActionError(null);
    try {
      await fn();
      setReload((r) => r + 1);
    } catch (e) {
      setActionError(String(e));
    } finally {
      setBusy(null);
    }
  }

  // Build nodes/edges and settle the layout ONCE (synchronously) so the graph
  // appears already arranged instead of animating from chaos on every open.
  const { nodes, edges, byId, layoutEdges } = useMemo(() => {
    const src = data?.nodes ?? [];
    const n = src.length;
    const ns: Node[] = src.map((node, i) => {
      // Seed on a circle for a stable, fast-settling start.
      const ang = (i / Math.max(1, n)) * Math.PI * 2;
      const rad = 60 + (n > 0 ? (i % 7) * 26 : 0);
      return {
        id: node.id,
        kind: node.kind,
        label: node.label,
        isChunk: node.kind === "doc_chunk",
        salience: node.salience,
        topicId: node.topic_id ?? null,
        x: W / 2 + Math.cos(ang) * rad,
        y: H / 2 + Math.sin(ang) * rad,
        vx: 0,
        vy: 0,
      };
    });
    // All relationships: part_of (doc->chunk), mentions (->entity), in_topic
    // (->topic), and related (memory<->memory similarity). "related" renders
    // faint so it shows the web without dominating.
    const es: Edge[] = (data?.edges ?? []).map((e) => ({
      a: e.from,
      b: e.to,
      relation: e.relation,
    }));
    // The layout is driven ONLY by structure (part_of / mentions / in_topic) so
    // it stays stable; "related" edges are drawn but do not pull nodes around.
    const layoutEdges = es.filter((e) => e.relation !== "related");
    const map = new Map(ns.map((x) => [x.id, x]));
    for (let k = 0; k < SETTLE_ITERS; k++) tick(ns, layoutEdges, map);
    return { nodes: ns, edges: es, byId: map, layoutEdges };
  }, [data]);

  const nodesRef = useRef<Node[]>(nodes);
  nodesRef.current = nodes;

  // Legend statistics: counts, connection breakdown, clusters, memory status.
  const stats = useMemo(() => {
    const srcNodes = data?.nodes ?? [];
    const srcEdges = data?.edges ?? [];
    const byKind: Record<string, number> = {};
    for (const n of srcNodes) byKind[n.kind] = (byKind[n.kind] || 0) + 1;
    const byRel: Record<string, number> = {};
    for (const e of srcEdges) byRel[e.relation] = (byRel[e.relation] || 0) + 1;

    // Connected components over the (non-related) structural graph.
    const adjm = new Map<string, string[]>();
    for (const n of srcNodes) adjm.set(n.id, []);
    for (const e of srcEdges) {
      adjm.get(e.from)?.push(e.to);
      adjm.get(e.to)?.push(e.from);
    }
    const visited = new Set<string>();
    let clusters = 0;
    for (const n of srcNodes) {
      if (visited.has(n.id)) continue;
      // ignore singletons for cluster count
      const stack = [n.id];
      let size = 0;
      while (stack.length) {
        const cur = stack.pop()!;
        if (visited.has(cur)) continue;
        visited.add(cur);
        size++;
        for (const nb of adjm.get(cur) ?? []) if (!visited.has(nb)) stack.push(nb);
      }
      if (size > 1) clusters++;
    }

    const now = Date.now();
    let recent = 0;
    let expiring = 0;
    for (const m of memories) {
      if (m.created_at && now - new Date(m.created_at).getTime() < 86400000) recent++;
      if (m.due_at && !m.done) {
        const d = new Date(m.due_at + "T00:00:00").getTime();
        if (Number.isFinite(d) && d >= now - 86400000 && d < now + 7 * 86400000) expiring++;
      }
    }

    const memoryKinds = ["note", "observation", "event", "measurement"];
    const memCount = memoryKinds.reduce((s, k) => s + (byKind[k] || 0), 0);
    return {
      memories: memCount,
      documents: byKind["document"] || 0,
      entities: byKind["entity"] || 0,
      summaries: byKind["summary"] || 0,
      chunks: byKind["doc_chunk"] || 0,
      connections: srcEdges.length,
      partOf: byRel["part_of"] || 0,
      mentions: byRel["mentions"] || 0,
      related: byRel["related"] || 0,
      inTopic: byRel["in_topic"] || 0,
      topics: byKind["topic"] || 0,
      clusters,
      recent,
      expiring,
      forgotten: data?.archived ?? 0,
    };
  }, [data, memories]);

  /** Write current node positions straight to the DOM (no React render). */
  function paint() {
    for (const n of nodesRef.current) {
      const g = gRefs.current.get(n.id);
      if (g) g.setAttribute("transform", `translate(${n.x},${n.y})`);
    }
    edges.forEach((e, i) => {
      const line = lineRefs.current[i];
      if (!line) return;
      const a = byId.get(e.a);
      const b = byId.get(e.b);
      if (!a || !b) return;
      line.setAttribute("x1", String(a.x));
      line.setAttribute("y1", String(a.y));
      line.setAttribute("x2", String(b.x));
      line.setAttribute("y2", String(b.y));
    });
  }

  /** Animation loop runs ONLY while dragging / cooling down. */
  function animate() {
    tick(nodesRef.current, layoutEdges, byId);
    paint();
    if (dragId.current || coolRef.current > 0) {
      coolRef.current = Math.max(0, coolRef.current - 1);
      rafRef.current = requestAnimationFrame(animate);
    } else {
      rafRef.current = 0;
    }
  }
  function kick() {
    coolRef.current = 40;
    if (!rafRef.current) rafRef.current = requestAnimationFrame(animate);
  }

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  // Scroll to zoom, anchored on the cursor.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const p = toSvg(e.clientX, e.clientY);
      setView((v) => {
        const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
        const newW = Math.max(W * 0.15, Math.min(W * 3.5, v.w * factor));
        const scale = newW / v.w;
        return {
          x: p.x - (p.x - v.x) * scale,
          y: p.y - (p.y - v.y) * scale,
          w: newW,
          h: v.h * scale,
        };
      });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [data]);

  function toSvg(clientX: number, clientY: number) {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const p = pt.matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  }

  function onMove(e: React.MouseEvent) {
    if (dragId.current) {
      const { x, y } = toSvg(e.clientX, e.clientY);
      const n = nodesRef.current.find((n) => n.id === dragId.current);
      if (n) {
        n.x = x;
        n.y = y;
        n.pinned = true;
      }
      return;
    }
    if (pan.current) {
      const a = toSvg(pan.current.x, pan.current.y);
      const b = toSvg(e.clientX, e.clientY);
      setView((v) => ({ ...v, x: v.x - (b.x - a.x), y: v.y - (b.y - a.y) }));
      pan.current = { x: e.clientX, y: e.clientY };
    }
  }

  function onBackgroundDown(e: React.MouseEvent) {
    if (e.target === svgRef.current) pan.current = { x: e.clientX, y: e.clientY };
  }
  function endInteract() {
    if (dragId.current) {
      dragId.current = null;
      kick(); // let neighbours settle briefly, then stop
    }
    pan.current = null;
  }

  const sel = selected ? memories.find((m) => m.id === selected) : null;
  const selTopic = selected ? topics.find((t) => t.id === selected) : undefined;
  const selNode = selected ? byId.get(selected) : undefined;
  const topicRows = [...topics].sort((a, b) => b.members - a.members);

  /** In topic-focus mode, is this node part of the focused subject? */
  const inFocus = (n: Node) => !focusTopic || n.id === focusTopic || n.topicId === focusTopic;
  const focusClass = (n: Node) => (focusTopic ? (inFocus(n) ? " hot" : " dim") : "");

  if (!data) return <div className="graph-empty muted">Loading memory graph…</div>;
  if (nodes.length === 0) {
    return (
      <div className="graph-empty muted">
        No memories yet. Chat with Mint or drop a document in the Vault — the graph grows
        as memories relate, entities connect them, and documents branch into chunks.
      </div>
    );
  }

  return (
    <div className="graph-wrap">
      <svg
        ref={svgRef}
        className="graph-svg"
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMid meet"
        onMouseDown={onBackgroundDown}
        onMouseMove={onMove}
        onMouseUp={endInteract}
        onMouseLeave={endInteract}
        onDoubleClick={() => setView({ x: 0, y: 0, w: W, h: H })}
      >
        {edges.map((e, i) => {
          const a = byId.get(e.a);
          const b = byId.get(e.b);
          if (!a || !b) return null;
          return (
            <line
              key={i}
              ref={(el) => {
                lineRefs.current[i] = el;
              }}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={`graph-edge ${e.relation}${
                focusTopic ? (inFocus(a) && inFocus(b) ? " hot" : " cold") : ""
              }`}
            />
          );
        })}
        {nodes.map((n) => (
          <g
            key={n.id}
            ref={(el) => {
              gRefs.current.set(n.id, el);
            }}
            transform={`translate(${n.x},${n.y})`}
            className={`graph-node ${n.kind} ${selected === n.id ? "sel" : ""}${focusClass(n)}`}
            onMouseDown={(e) => {
              e.preventDefault();
              dragId.current = n.id;
              setSelected(n.id);
              kick();
            }}
          >
            <circle r={radius(n, selected === n.id)} />
            {!n.isChunk && (
              <text x={radius(n, selected === n.id) + 4} y={4}>
                {n.label}
              </text>
            )}
          </g>
        ))}
      </svg>

      <div className="graph-panel">
        <div className="gp-section">
          <h4>Statistics</h4>
          <div className="gp-row"><i className="lg note" /><span>Memories</span><b>{stats.memories}</b></div>
          <div className="gp-row"><i className="lg document" /><span>Documents</span><b>{stats.documents}</b></div>
          <div className="gp-row"><i className="lg entity" /><span>Entities</span><b>{stats.entities}</b></div>
          <div className="gp-row"><i className="lg summary" /><span>Summaries</span><b>{stats.summaries}</b></div>
          <div className="gp-row"><i className="lg chunk" /><span>Chunks</span><b>{stats.chunks}</b></div>
          <div className="gp-row"><i className="lg topic" /><span>Topics</span><b>{stats.topics}</b></div>
        </div>
        <div className="gp-section">
          <h4>Topics</h4>
          {topicRows.length === 0 && <div className="gp-empty">No topics yet</div>}
          {topicRows.map((t) => (
            <button
              key={t.id}
              className={`gp-row gp-topic ${focusTopic === t.id ? "on" : ""}`}
              title={t.summary || t.name}
              onClick={() => setFocusTopic(focusTopic === t.id ? null : t.id)}
            >
              <i className="lg topic" />
              <span>{t.name}</span>
              <b>{t.members}</b>
            </button>
          ))}
          {focusTopic && (
            <button className="ghost gp-action" onClick={() => setFocusTopic(null)}>
              Show all topics
            </button>
          )}
          <button
            className="ghost gp-action"
            disabled={!!busy}
            onClick={() => act("organize", organizeTopics)}
            title="File every not-yet-organized memory and document into topics, then summarize them"
          >
            {busy === "organize" ? "Organizing..." : "Organize memories"}
          </button>
        </div>
        <div className="gp-section">
          <h4>Connections <b className="gp-total">{stats.connections}</b></h4>
          <div className="gp-row"><i className="lg c-part" /><span>Document source</span><b>{stats.partOf}</b></div>
          <div className="gp-row"><i className="lg c-mention" /><span>Entity mentions</span><b>{stats.mentions}</b></div>
          <div className="gp-row"><i className="lg c-topic" /><span>Topic membership</span><b>{stats.inTopic}</b></div>
          <div className="gp-row"><i className="lg c-related" /><span>Related</span><b>{stats.related}</b></div>
        </div>
        <div className="gp-section">
          <h4>Clusters</h4>
          <div className="gp-row"><span>Visible clusters</span><b>{stats.clusters}</b></div>
        </div>
        <div className="gp-section">
          <h4>Memory status</h4>
          <div className="gp-row"><i className="dot-stat recent" /><span>Recent (&lt; 24h)</span><b>{stats.recent}</b></div>
          <div className="gp-row"><i className="dot-stat expiring" /><span>Expiring soon</span><b>{stats.expiring}</b></div>
          <div className="gp-row"><i className="dot-stat forgotten" /><span>Forgotten (archived)</span><b>{stats.forgotten}</b></div>
        </div>
        {actionError && !selected && <div className="gp-error">{actionError}</div>}
        <div className="gp-hint">scroll to zoom · drag to pan · double-click to reset</div>
      </div>

      {selTopic && (
        <div className="graph-detail">
          <div className="gd-head">
            <span className="pill topic">topic</span>
            <strong>{selTopic.name}</strong>
            <button className="ghost" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>
          <p className="gd-text">
            {selTopic.summary || "No summary yet. Refresh to distill one from its members."}
          </p>
          <div className="gd-meta">
            <span>{selTopic.members} members</span>
            {selTopic.user_named && <span>named by you</span>}
          </div>
          <div className="gd-actions">
            <div className="gd-field">
              <input
                value={renameVal}
                placeholder="Rename topic"
                onChange={(e) => setRenameVal(e.target.value)}
              />
              <button
                className="ghost"
                disabled={!renameVal.trim() || !!busy}
                onClick={() => act("rename", () => renameTopic(selTopic.id, renameVal.trim()))}
              >
                Rename
              </button>
            </div>
            <div className="gd-field">
              <select value={mergeTarget} onChange={(e) => setMergeTarget(e.target.value)}>
                <option value="">Merge into...</option>
                {topicRows
                  .filter((t) => t.id !== selTopic.id)
                  .map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
              </select>
              <button
                className="ghost"
                disabled={!mergeTarget || !!busy}
                onClick={() =>
                  act("merge", async () => {
                    await mergeTopics(selTopic.id, mergeTarget);
                    if (focusTopic === selTopic.id) setFocusTopic(mergeTarget);
                    setSelected(mergeTarget);
                  })
                }
              >
                Merge
              </button>
            </div>
            <div className="gd-field">
              <button className="ghost" onClick={() => setFocusTopic(selTopic.id)}>
                Focus
              </button>
              <button
                className="ghost"
                disabled={!!busy}
                onClick={() => act("refresh", () => refreshTopic(selTopic.id))}
              >
                {busy === "refresh" ? "Summarizing..." : "Refresh summary"}
              </button>
            </div>
            {actionError && <div className="gp-error">{actionError}</div>}
          </div>
        </div>
      )}

      {sel && !selTopic && (
        <div className="graph-detail">
          <div className="gd-head">
            <span className={`pill ${sel.kind}`}>{sel.kind}</span>
            <strong>{sel.title || "(untitled)"}</strong>
            <button className="ghost" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>
          <p className="gd-text">{sel.text}</p>
          <div className="gd-meta">
            {sel.tags.map((t) => (
              <span key={t} className="tag">
                #{t}
              </span>
            ))}
            {sel.source && <span>source: {sel.source}</span>}
            <span className={`sync ${sel.sync_state}`}>{sel.sync_state}</span>
          </div>
          {sel.kind !== "doc_chunk" && sel.kind !== "entity" && topicRows.length > 0 && (
            <div className="gd-actions">
              <div className="gd-field">
                <span className="gd-label">Topic</span>
                <select
                  value={selNode?.topicId ?? ""}
                  disabled={!!busy}
                  onChange={(e) =>
                    e.target.value && act("move", () => moveToTopic(sel.id, e.target.value))
                  }
                >
                  <option value="">Unfiled</option>
                  {topicRows.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </div>
              {actionError && <div className="gp-error">{actionError}</div>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
