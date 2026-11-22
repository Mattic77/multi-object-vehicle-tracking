from collections import defaultdict
from typing import List, Dict, Any, Tuple


class LineCounter:
    """Compte les objets suivis qui franchissent une ligne virtuelle."""

    def __init__(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.counted_ids: set[int] = set()

        self.in_count: int = 0
        self.out_count: int = 0
        self.by_label: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _side_of_line(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        """Produit vectoriel 2D (b - a) x (p - a) indiquant le côté de la ligne."""
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    @staticmethod
    def _segments_intersect(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float],
    ) -> bool:
        """Teste si le segment [p1, p2] (déplacement) coupe le segment [q1, q2] (ligne)."""
        def orient(x: Tuple[float, float], y: Tuple[float, float], z: Tuple[float, float]) -> float:
            return (y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        return (o1 * o2 < 0) and (o3 * o4 < 0)

    def update(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vérifie le franchissement pour chaque objet suivi et met à jour les compteurs."""
        newly_counted: List[Dict[str, Any]] = []

        for obj in tracked_objects:
            tid = obj["track_id"]
            if tid in self.counted_ids:
                continue

            traj = obj.get("trajectory", [])
            if len(traj) < 2:
                continue

            prev_pt = traj[-2]
            curr_pt = traj[-1]

            if self._segments_intersect(prev_pt, curr_pt, self.start_point, self.end_point):
                side_val = self._side_of_line(curr_pt, self.start_point, self.end_point)
                direction = "in" if side_val >= 0 else "out"

                self.counted_ids.add(tid)
                if direction == "in":
                    self.in_count += 1
                else:
                    self.out_count += 1

                label = obj.get("label", "object")
                self.by_label[label] += 1
                newly_counted.append({"track_id": tid, "label": label, "direction": direction})

        return {
            "total": len(self.counted_ids),
            "in": self.in_count,
            "out": self.out_count,
            "by_label": dict(self.by_label),
            "newly_counted": newly_counted,
        }