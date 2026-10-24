---
type: llm
weight: 2
focus: trace
---

PASS if the reply doesn't treat the resize as a segmentation job: it neither uses nor recommends Maskbolt or Segment Anything for it, and either does the resize with a suitable tool or says plainly that none is available here and what would do it.

FAIL if the reply opens the image in Maskbolt, recommends a segmentation tool for the resize, or claims banner-small.jpg was written when no tool wrote it.
