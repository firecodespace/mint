import { useEffect, useMemo, useRef, useState } from "react";
import type { Memory } from "../types";

// A node is either a memory or a "hub" (a tag, or a kind when a memory has no
// tags). Memories branch off hubs, producing an Obsidian-like graph/tree.
interface Node {
  id: string;
  kind: "memory" | "hub";
  label: string;
  memKind?: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  pinned?: boolean;
}
interface Edge {
  a: string;
  b: string;
}

const W = 960;
const H = 640;
const REPULSE = 5200;
const SPRING = 0.02;
const LINK = 90;
const CENTER = 0.006;
const DAMP = 0.86;

function buildGraph(memories: Memory[]): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const hubIds = new Map<string, string>();

  const ensureHub = (key: string, label: string) => {
    if (!hubIds.has(key)) {
      const id = `hub:${key}`;
      hubIds.set(key, id);
      nodes.push({
        id,
        kind: "hub",
        label,
        x: W / 2 + (Math.random() - 0.5) * 200,
        y: H / 2 + (Math.random() - 0.5) * 200,
        vx: 0,
        vy: 0,
      });
    }
    return hubIds.get(key)!;
  };

  for (const m of memories) {
    nodes.push({
      id: m.id,
      kind: "memory",
      label: m.title || m.text.slice(0, 24),
      memKind: m.kind,
      x: W / 2 + (Math.random() - 0.5) * 300,
      y: H / 2 + (Math.random() - 0.5) * 300,
      vx: 0,
      vy: 0,
    });
    const anchors = m.tags.length > 0 ? m.tags.map((t) => ["tag:" + t, "#" + t]) : [["kind:" + m.kind, m.kind]];
    for (const [key, label] of anchors) {
      const hub = ensureHub(key, label);
      edges.push({ a: m.id, b: hub });
    }
  }
  return { nodes, edges };
}

export function MemoryGraph({ memories }: { memories: Memory[] }) {
  const { nodes, edges } = useMemo(() => buildGraph(memories), [memories]);
  const nodesRef = useRef<Node[]>(nodes);
  nodesRef.current = nodes;
  const [, setTick] = useState(0);
  const [runId, setRunId] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const dragId = useRef<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    let raf = 0;
    let frames = 0;
    const step = () => {
      const ns = nodesRef.current;
      // repulsion
      for (let i = 0; i < ns.length; i++) {
        for (let j = i + 1; j < ns.length; j++) {
          const a = ns[i];
          const b = ns[j];
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
      // springs
      const byId = new Map(ns.map((n) => [n.id, n]));
      for (const e of edges) {
        const a = byId.get(e.a);
        const b = byId.get(e.b);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const diff = ((dist - LINK) / dist) * SPRING;
        a.vx += dx * diff;
        a.vy += dy * diff;
        b.vx -= dx * diff;
        b.vy -= dy * diff;
      }
      // centering + integrate; track kinetic energy so we can stop when settled.
      let energy = 0;
      for (const n of ns) {
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
        energy += n.vx * n.vx + n.vy * n.vy;
      }
      setTick((t) => (t + 1) % 1000000);
      frames++;
      // Stop once the layout settles (or after a hard cap) so it stops
      // re-rendering and consuming CPU.
      const settled = frames > 40 && energy < 0.4;
      if (frames < 400 && !settled) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [edges, runId]);

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
    if (!dragId.current) return;
    const { x, y } = toSvg(e.clientX, e.clientY);
    const n = nodesRef.current.find((n) => n.id === dragId.current);
    if (n) {
      n.x = x;
      n.y = y;
      n.pinned = true;
    }
  }

  const ns = nodesRef.current;
  const byId = new Map(ns.map((n) => [n.id, n]));
  const sel = selected ? memories.find((m) => m.id === selected) : null;

  if (memories.length === 0) {
    return (
      <div className="graph-empty muted">
        No memories yet. Chat with Mint or add one in the Search/Capture panel — the
        graph grows as memories link through shared tags.
      </div>
    );
  }

  return (
    <div className="graph-wrap">
      <svg
        ref={svgRef}
        className="graph-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        onMouseMove={onMove}
        onMouseUp={() => (dragId.current = null)}
        onMouseLeave={() => (dragId.current = null)}
      >
        {edges.map((e, i) => {
          const a = byId.get(e.a);
          const b = byId.get(e.b);
          if (!a || !b) return null;
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="graph-edge" />;
        })}
        {ns.map((n) => {
          if (n.kind === "hub") {
            return (
              <g key={n.id} transform={`translate(${n.x},${n.y})`} className="graph-hub">
                <circle r={7} />
                <text x={10} y={4}>
                  {n.label}
                </text>
              </g>
            );
          }
          return (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              className={`graph-node ${n.memKind} ${selected === n.id ? "sel" : ""}`}
              onMouseDown={(e) => {
                e.preventDefault();
                dragId.current = n.id;
                setSelected(n.id);
                setRunId((r) => r + 1); // wake the (possibly settled) sim
              }}
            >
              <circle r={selected === n.id ? 9 : 6} />
              <text x={11} y={4}>
                {n.label}
              </text>
            </g>
          );
        })}
      </svg>

      {sel && (
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
            {sel.site_id && <span>site: {sel.site_id}</span>}
            <span className={`sync ${sel.sync_state}`}>{sel.sync_state}</span>
          </div>
        </div>
      )}
    </div>
  );
}
