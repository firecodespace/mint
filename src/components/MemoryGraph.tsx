import { useEffect, useMemo, useRef, useState } from "react";
import { graphData } from "../api";
import type { GraphData, Memory } from "../types";

interface Node {
  id: string;
  kind: string;
  label: string;
  isChunk: boolean;
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

const W = 960;
const H = 640;
const REPULSE = 4200;
const SPRING = 0.03;
const LINK_PART = 55; // chunk -> document (tight cluster)
const LINK_REL = 120; // related memories (looser)
const CENTER = 0.006;
const DAMP = 0.86;

function radius(n: Node, selected: boolean) {
  if (selected) return n.kind === "document" ? 11 : 9;
  if (n.kind === "document") return 9;
  if (n.isChunk) return 3.5;
  return 6;
}

export function MemoryGraph({ memories }: { memories: Memory[] }) {
  const [data, setData] = useState<GraphData | null>(null);
  const [, setTick] = useState(0);
  const [runId, setRunId] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const dragId = useRef<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    graphData().then(setData).catch(console.error);
  }, [memories.length]);

  const { nodes, edges } = useMemo(() => {
    const ns: Node[] = (data?.nodes ?? []).map((n) => ({
      id: n.id,
      kind: n.kind,
      label: n.label,
      isChunk: n.kind === "doc_chunk",
      x: W / 2 + (Math.random() - 0.5) * 320,
      y: H / 2 + (Math.random() - 0.5) * 320,
      vx: 0,
      vy: 0,
    }));
    const es: Edge[] = (data?.edges ?? []).map((e) => ({
      a: e.from,
      b: e.to,
      relation: e.relation,
    }));
    return { nodes: ns, edges: es };
  }, [data]);

  const nodesRef = useRef<Node[]>(nodes);
  nodesRef.current = nodes;

  useEffect(() => {
    let raf = 0;
    let frames = 0;
    const byId = new Map(nodesRef.current.map((n) => [n.id, n]));
    const step = () => {
      const ns = nodesRef.current;
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
          a.vx += (dx / d) * f;
          a.vy += (dy / d) * f;
          b.vx -= (dx / d) * f;
          b.vy -= (dy / d) * f;
        }
      }
      for (const e of edges) {
        const a = byId.get(e.a);
        const b = byId.get(e.b);
        if (!a || !b) continue;
        const target = e.relation === "part_of" ? LINK_PART : LINK_REL;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const diff = ((dist - target) / dist) * SPRING;
        a.vx += dx * diff;
        a.vy += dy * diff;
        b.vx -= dx * diff;
        b.vy -= dy * diff;
      }
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
      const settled = frames > 50 && energy < 0.4;
      if (frames < 500 && !settled) raf = requestAnimationFrame(step);
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

  if (!data) {
    return <div className="graph-empty muted">Loading memory graph…</div>;
  }
  if (ns.length === 0) {
    return (
      <div className="graph-empty muted">
        No memories yet. Chat with Mint or drop a document in the Vault — the graph grows
        as memories relate and documents branch into chunks.
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
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={`graph-edge ${e.relation}`}
            />
          );
        })}
        {ns.map((n) => (
          <g
            key={n.id}
            transform={`translate(${n.x},${n.y})`}
            className={`graph-node ${n.kind} ${selected === n.id ? "sel" : ""}`}
            onMouseDown={(e) => {
              e.preventDefault();
              dragId.current = n.id;
              setSelected(n.id);
              setRunId((r) => r + 1);
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

      <div className="graph-legend">
        <span><i className="lg document" /> document</span>
        <span><i className="lg note" /> memory</span>
        <span><i className="lg chunk" /> chunk</span>
        <span><i className="lg e-part" /> part of</span>
        <span><i className="lg e-rel" /> related</span>
      </div>

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
            {sel.source && <span>source: {sel.source}</span>}
            <span className={`sync ${sel.sync_state}`}>{sel.sync_state}</span>
          </div>
        </div>
      )}
    </div>
  );
}
