---
title: Healthcare Claims Cost Intelligence
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Healthcare Claims Cost Intelligence

Predicts the cost tier and total charge of a Medicare-style claim, with SHAP
feature attributions behind every prediction.

The dashboard is the page you are looking at. A FastAPI service runs alongside
it inside the same container and serves the predictions.

Source: https://github.com/ManojMareedu/Health-Care-Project
