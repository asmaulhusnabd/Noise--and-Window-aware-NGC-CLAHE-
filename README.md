# Noise- & Window-Aware NGC-CLAHE for CT
- Reproduces NGC-CLAHE and adds window-aware + noise/edge-aware blending.
- Eval metrics: UIQI, SSIM, FSIM (as in base paper).

# How to run this project?
- Create a virtual environment based on requirement.text
- Activate the virtual environment
- Now run the following commands:
### 1.  To make synthetic degraded low contrast image
- `python -m src.run_make_synth --src data/real --dst data/synth --mode soft`
### 2. To run the methods (CLAHE, NGC-CLAHE, Our method) on the synthetic low contrast image
- `python -m src.run_methods   --src data/synth --out data/outputs --mode soft`
### 3. To get the output on three different metrices
- `python -m src.run_metrics   --ref data/real --out data/outputs --mode soft`

### 4. To do visualization
- `python notebooks/preview_best.py`
