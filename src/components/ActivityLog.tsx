export interface ActivityEvent {
  id: number;
  time: Date;
  kind: "system" | "ingest" | "search" | "delete" | "error";
  message: string;
}

const ICONS: Record<ActivityEvent["kind"], string> = {
  system: "◍",
  ingest: "＋",
  search: "⌕",
  delete: "－",
  error: "⚠",
};

export function ActivityLog({ events }: { events: ActivityEvent[] }) {
  return (
    <div className="activity-inner">
      <h3>Activity</h3>
      <ul className="event-list">
        {events.length === 0 && <li className="muted">No activity yet.</li>}
        {events.map((e) => (
          <li key={e.id} className={`event ${e.kind}`}>
            <span className="ic">{ICONS[e.kind]}</span>
            <span className="msg">{e.message}</span>
            <span className="ts">
              {e.time.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
