---
type: llm
weight: 2
focus: trace
---

PASS if the trace shows photos/desk.jpg opened, a kept mask that is the mug with its handle alone (the middle level, area about 27,400, bbox [150, 250, 345, 430]; not the larger mask that takes in the phone, and not just the coffee surface), and a cutout export of that mask to photos/mug.png that succeeded, and the final reply gives that path.

FAIL if the exported cutout is the mug-and-phone mask or the coffee surface, the reply claims an export the trace doesn't show succeeding, it overwrites photos/desk.jpg, or it only explains how to do the cutout.
