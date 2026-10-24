---
description: A dense image of small objects needs a tiled find before every kernel has a mask.
max_turns: 30
timeout_seconds: 600
runs: 1
tags: [maskbolt, find, tiles]
allowed_tools: [Skill]
---

I need a mask for every kernel on the corn cob in corn/cob.png. They're small and packed tight, so don't stop at the obvious ones. Save a labels image to corn/kernels-labels.png and the outlines as JSON to corn/kernels.json, and tell me how many kernels you got.
