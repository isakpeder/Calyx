# Vision calibration benchmark

Measures how accurately the pipeline reports wound area in cm², and how much
of that accuracy comes from the reference-coin calibration.

| File | Purpose |
|---|---|
| `scenes.py` | Renders synthetic wound photos with exact ground-truth area: irregular wound with slough patches, a US quarter at the correct size, exposure / noise / blur. |
| `run_calibration.py` | Measures area three ways per scene and writes `evals/results/vision_calibration_<label>.json`. |

```bash
python -m evals.vision.run_calibration --label <name>   # 200 seeded scenes
```

## Conditions

- **coin**: full pipeline (`analyze_frame`) on the photo with a coin.
- **fallback**: same photo without the coin, so the pipeline uses its fixed 0.026 cm/px guess.
- **oracle**: wound mask with the true scale. Shows segmentation error alone.

Simulated camera resolution is log-uniform from 19 to 77 px/cm, which is half to
double what the fallback assumes. **The fallback error depends directly on this
range** (a wider range gives a worse fallback), so treat it as illustrative.
The coin and oracle conditions do not depend on it.

Five skin tones, light to dark, are sampled uniformly and reported separately.

## Metrics

- **median error**: median of |measured − true| / true.
- **within 10%**: share of scans measured within 10% of the true area.
- **accuracy**: 100 − mean % error, floored at 0.

## Limitations

The images are synthetic: flat lighting, uniform skin texture, and an idealized coin.
Real photos are harder. Confirming these numbers needs a physical test: paper
cutouts of known area photographed beside a quarter at several distances.
