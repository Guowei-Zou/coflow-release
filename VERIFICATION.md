# Release verification — 2026-09-30

A fresh Python 3.8 environment installed requirements.txt successfully and passed pip check. The actual policy training launcher completed two optimizer updates on a 100-episode subset of real Spread Medium data, produced checkpoints, and the saved policy was loaded for real MPE evaluation by the downstream MA-WAM and G2MAF entry points. The installation self-check exercises MPE reset and step without datasets.

These are installation and short functional checks, not full-budget retraining or reproduction of paper scores. They do not certify every map, data split, historical checkpoint, optional rendering path, or inherited prototype. Simulator binaries/maps and datasets remain external requirements. Training seeds and evaluation seeds are distinct.
