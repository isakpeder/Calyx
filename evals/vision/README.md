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

## Results

200 scenes, all five skin tones. "Original" is the hackathon pipeline.

| Coin condition | Original | + coin radius refinement | + skin-relative mask |
|---|---|---|---|
| Median area error | 100% | 100% | **1.9%** |
| Scans within 10% of true area | 10.0% | 39.5% | **97.0%** |
| Accuracy (100 − mean % error) | 32.6% | 41.3% | **96.4%** |
| Coin radius error (median) | 8.6% | 0.3% | 0.3% |

The no-coin fallback reaches 18.9% accuracy (6.0% of scans within 10%) on
the final pipeline.

By skin tone, scans within 10% with the coin:

| Tone | Original | Final |
|---|---|---|
| light | 29.5% | 90.9% |
| fair | 21.2% | 97.0% |
| medium | 0.0% | 97.9% |
| tan | 0.0% | 100.0% |
| dark | 0.0% | 100.0% |

The original pipeline failed on medium, tan and dark skin because its fixed
HSV thresholds marked skin as wound. The mask now excludes pixels close to
the patient's own skin colour (CIELAB ΔE < 25). Both fixes were developed on
scene seeds 50000+; these results use the benchmark seeds 10000+.

## Limitations

The images are synthetic: flat lighting, uniform skin texture, an idealized
coin, and wound colours that contrast strongly with skin. Real wounds on darker
skin can be lower contrast.
Real photos are harder. Confirming these numbers needs a physical test: paper
cutouts of known area photographed beside a quarter at several distances.
