---
title: Crucible
emoji: 🔥
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Crucible — Space

An in-browser taste of [Crucible](https://github.com/nadeauglenn1-max/crucible): turn
real software into a trainable, gradable, **replayable** RL environment for AI agents.

Pick a scenario, run a scripted agent through the environment, and watch the recorded
episode reproduce byte-for-byte. All the logic is the tested `crucible` core; `app.py`
only wires it to Gradio.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy this Space

From this `space/` directory, once you're logged in to Hugging Face
(`huggingface-cli login`):

```bash
huggingface-cli repo create crucible --type space --space_sdk gradio
git init && git add . && git commit -m "Crucible Space"
git remote add hf https://huggingface.co/spaces/<your-username>/crucible
git push hf main
```
