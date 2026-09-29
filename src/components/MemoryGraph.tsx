import { useEffect, useMemo, useRef, useState } from "react";
import { graphData } from "../api";
import type { GraphData, Memory } from "../types";

interface Node {
  id: string;
  kind: string;
  label: string;
  isChunk: boolean;
  salience: number;
  cr: number; // collision radius
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

const CX = 500;
const CY = 360;
const REPULSE = 900;
const COLLIDE_PAD = 6;
const LINK_PART = 26;
const LINK_MENTION = 66;
const CENTER = 0.025;
const DAMP = 0.86;
const VMAX = 10;
const ITERS = 340;

function baseRadius(kind: string, salience: number, isChunk: boolean) {
  if (isChunk) return 3;
  const base = kind === "document" ? 9 : kind === "entity" || kind === "summary" ? 8 : 6;
  return base * (0.7 + 0.6 * (salience || 0.5));
}

/** One physics step: repulsion + collision + links + centering (Obsidian-style). */
function tick(nodes: Node[], edges: Edge[], byId: Map<string, Node>) {
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      let dx = a.x - b.x;
      let dy = a.y - b.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 0.01) {
        dx = Math.random() - 0.5;
        dy = Math.random() - 0.5;
        d2 = 1;
      }
      const d = Math.sqrt(d2);
      // charge repulsion (soft, capped so it never explodes)
      const f = Math.min(REPULSE / d2, 4);
      a.vx += (dx / d) * f;
      a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f;
      b.vy -= (dy / d) * f;
      // collision: hard-ish separation so nodes never overlap
      const minD = a.cr + b.cr + COLLIDE_PAD;
      if (d < minD) {
        const push = (minD - d) * 0.5;
        a.vx += (dx / d) * push;
        a.vy += (dy / d) * push;
        b.vx -= (dx / d) * push;
        b.vy -= (dy / d) * push;
      }
    }
  }
  for (const e of edges) {
    const a = byId.get(e.a);
    const b = byId.get(e.b);
    if (!a || !b) continue;
    const len = e.relation === "part_of" ? LINK_PART : LINK_MENTION;
    const k = e.relation === "part_of" ? 0.09 : 0.05;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const diff = ((dist - len) / dist) * k;
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
    n.vx += (CX - n.x) * CENTER;
    n.vy += (CY - n.y) * CENTER;
    n.vx *= DAMP;
    n.vy *= DAMP;
    const sp = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
    if (sp > VMAX) {
      n.vx = (n.vx / sp) * VMAX;
      n.vy = (n.vy / sp) * VMAX;
    }
    n.x += n.vx;
    n.y += n.vy;
  }
}

export function MemoryGraph({ memories }: { memories: Memory[] }) {
  const [data, setData] = useState<GraphData | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, w: 1000, h: 720 });
  const [selected, setSelected] = useState<string | null>(null);
  const dragId = useRef<string | null>(null);
  const pan = useRef<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const gRefs = useRef<Map<string, SVGGElement | null>>(new Map());
  const lineRefs = useRef<(SVGLineElement | null)[]>([]);
  const rafRef = useRef(0);
  const coolRef = useRef(0);

  useEffect(() => {
    graphData().then(setData).catch(console.error);
  }, [memories.length]);

  const { nodes, edges, byId } = useMemo(() => {
    const src = data?.nodes ?? [];
    const nodeIds = new Set(src.map((n) => n.id));
    const es: Edge[] = (data?.edges ?? [])
      .filter((e) => e.relation !== "related" && nodeIds.has(e.from) && nodeIds.has(e.to))
      .map((e) => ({ a: e.from, b: e.to, relation: e.relation }));

    const n = src.length;
    const ns: Node[] = src.map((node, i) => {
      const isChunk = node.kind === "doc_chunk";
      const ang = (i / Math.max(1, n)) * Math.PI * 2;
      const seed = 90 + (i % 11) * 22;
      return {
        id: node.id,
        kind: node.kind,
        label: node.label,
        isChunk,
        salience: node.salience,
        cr: baseRadius(node.kind, node.salience, isChunk),
        x: CX + Math.cos(ang) * seed,
        y: CY + Math.sin(ang) * seed,
        vx: 0,
        vy: 0,
      };
    });
    const map = new Map(ns.map((x) => [x.id, x]));
    for (let k = 0; k < ITERS; k++) tick(ns, es, map);
    return { nodes: ns, edges: es, byId: map };
  }, [data]);

  const nodesRef = useRef<Node[]>(nodes);
  nodesRef.current = nodes;

  const adj = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of edges) {
      (m.get(e.a) ?? m.set(e.a, new Set()).get(e.a)!).add(e.b);
      (m.get(e.b) ?? m.set(e.b, new Set()).get(e.b)!).add(e.a);
    }
    return m;
  }, [edges]);
  const neighbors = selected ? adj.get(selected) ?? new Set<string>() : null;

  function fit() {
    const ns = nodesRef.current;
    if (ns.length === 0) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const nd of ns) {
      if (nd.x < minX) minX = nd.x;
      if (nd.y < minY) minY = nd.y;
      if (nd.x > maxX) maxX = nd.x;
      if (nd.y > maxY) maxY = nd.y;
    }
    const pad = 50;
    setView({
      x: minX - pad,
      y: minY - pad,
      w: Math.max(maxX - minX + 2 * pad, 300),
      h: Math.max(maxY - minY + 2 * pad, 200),
    });
  }

  useEffect(() => {
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes]);

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

  function animate() {
    tick(nodesRef.current, edges, byId);
    paint();
    if (dragId.current || coolRef.current > 0) {
      coolRef.current = Math.max(0, coolRef.current - 1);
      rafRef.current = requestAnimationFrame(animate);
    } else {
      rafRef.current = 0;
    }
  }
  function kick() {
    coolRef.current = 45;
    if (!rafRef.current) rafRef.current = requestAnimationFrame(animate);
  }

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const p = toSvg(e.clientX, e.clientY);
      setView((v) => {
        const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
        const newW = Math.max(150, Math.min(4000, v.w * factor));
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
      const n = byId.get(dragId.current);
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
    if (e.target === svgRef.current) {
      pan.current = { x: e.clientX, y: e.clientY };
      setSelected(null);
    }
  }
  function endInteract() {
    if (dragId.current) {
      dragId.current = null;
      kick();
    }
    pan.current = null;
  }

  const sel = selected ? memories.find((m) => m.id === selected) : null;

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
        onDoubleClick={fit}
      >
        {edges.map((e, i) => {
          const a = byId.get(e.a);
          const b = byId.get(e.b);
          if (!a || !b) return null;
          const state = !selected
            ? ""
            : e.a === selected || e.b === selected
            ? "hot"
            : "cold";
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
              className={`graph-edge ${e.relation} ${state}`}
            />
          );
        })}
        {nodes.map((n) => {
          const nstate = !selected
            ? ""
            : n.id === selected
            ? "sel"
            : neighbors?.has(n.id)
            ? "near"
            : "far";
          const r = baseRadius(n.kind, n.salience, n.isChunk) + (selected === n.id ? 3 : 0);
          return (
            <g
              key={n.id}
              ref={(el) => {
                gRefs.current.set(n.id, el);
              }}
              transform={`translate(${n.x},${n.y})`}
              className={`graph-node ${n.kind} ${nstate}`}
              onMouseDown={(e) => {
                e.preventDefault();
                dragId.current = n.id;
                setSelected(n.id);
                kick();
              }}
            >
              <circle r={r} />
              {!n.isChunk && (
                <text x={r + 4} y={4}>
                  {n.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <div className="graph-legend">
        <span><i className="lg document" /> document</span>
        <span><i className="lg entity" /> entity</span>
        <span><i className="lg summary" /> summary</span>
        <span><i className="lg note" /> memory</span>
        <span><i className="lg chunk" /> chunk</span>
        <span className="lg-hint">click a node to focus · scroll to zoom · drag to pan · double-click to fit</span>
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
