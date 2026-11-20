# Analysis Plan — Smart Insole Sensor Placement (BME 411, Group 9)

Two Python files: `data.py` prepares the data once, `optimize.py` runs the optimizers and produces the report outputs.

## File 1: `data.py` — prepare the data once

1. **Sensor locations.** Parse `sensors_map_left.svg` and `sensors_map_right.svg` to get the (x, y) positions of the 32 real sensors on the insole.
2. **Load walking trials.** For each Normal Pace (NP) trial, read the left/right insole pressures (64 Hz) and the GAITRite ground-truth heel-strike and toe-off times. Use `sync_auto.json` to put both on the same clock.
3. **Continuous pressure map.** At every time frame, interpolate the 32 real sensors onto a dense grid over the insole, so a virtual sensor placed anywhere (e.g. 4.2 cm, 11.7 cm) has a pressure reading.
4. **Score a layout.** Given sensor positions:
   - Sum their pressures into one signal.
   - Detect heel-strike (signal rises past a threshold) and toe-off (signal falls below it).
   - Compute step and stride times.
   - Compare to GAITRite and return the error E (RMSE, ms).
   - Add a penalty if sensors overlap.
5. **Cache.** Save the preprocessed data to disk so later runs load in seconds.

## File 2: `optimize.py` — run the optimizers and generate outputs

6. **MOPSO (main method).** Each particle = N + 16 (x, y) pairs, using only the first N pairs. The swarm minimizes error and sensor count simultaneously. An external archive keeps the best error found for each N; that archive is the Pareto front.
7. **PSO sweep (check).** For each N = 2…16, run a standard single-objective PSO on error. Its error-per-N curve should match the MOPSO front.
8. **SLSQP comparison.** Run SciPy's gradient-based solver from multiple random starts for N = 4 and N = 6 to show local-minimum trapping and non-convexity.
9. **Pick the optimum.** Apply f = W₁·E + W₂·N to the front across several weight choices to find the elbow point.
10. **Cross-cadence validation.** Take the optimal layouts (trained on NP only) and score them on Slow Pace (SP) and Fast Pace (FP) trials without re-optimizing.
11. **Export deliverables:**
    - `pareto_front.png`: error vs. sensor count
    - `optimal_layouts.png`: insole with pressure heatmap and optimal sensors for N = 3, 4, 5, 8
    - `convergence_comparison.png`: MOPSO/PSO vs. multi-start SLSQP
    - `cross_cadence_validation.csv`: NP-trained layouts tested on SP and FP
