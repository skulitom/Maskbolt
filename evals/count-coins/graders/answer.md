---
type: llm
weight: 2
focus: trace
---

PASS if the trace shows a find on tray/coins.png, the two coins it left uncovered then added (clicked, or maskbolt_gaps with add: true), and an SVG export of all 26 masks to tray/coins.svg with mm_per_px 0.2 that succeeded, and the final reply says there are 26 coins and gives the SVG's path.

FAIL if the reply gives 24 (the count before the uncovered patches were handled) or any count the trace doesn't support, the export lacks mm_per_px 0.2, the reply claims an export the trace doesn't show succeeding, or it counts by guessing instead of segmenting.
