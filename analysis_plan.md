# Analysis Plan — Smart Insole Sensor Placement (BME 411, Group 9)

Two Python files: `data.py` prepares the data once, `optimize.py` runs the optimizers and produces the report outputs.

## File 1: `data.py` — prepare the data once

1. **Sensor locations.** Parse `sensors_map_left.svg` and `sensors_map_right.svg` to get the (x, y) positions of the 32 real sensors on the insole.
2. **Load walking trials.** For each Normal Pace (NP) trial, read the left/right insole pressures (64 Hz) and the GAITRite ground-truth heel-strike and toe-off times. Use `sync_auto.json` to put both on the same clock.
3. **Continuous pressure map.** At every time frame, interpolate the 32 real sensors onto a dense grid over the insole, so a virtual sensor placed anywhere (e.g. 4.2 cm, 11.7 cm) has a pressure reading.
4. **Score a layout.** Given sensor positions:
   - Combine their pressures into one contact signal. Two methods are run and compared:
     - **Summed signal:** add all sensor pressures together. Heel-strike is when the total rises past 10% of its range, and toe-off is when it falls below.
     - **First/last sensor:** heel-strike is when the *first* sensor loads, and toe-off is when the *last* sensor unloads. A sensor counts as loaded at 30% of the strongest sensor's range. The threshold is higher than 10% because single sensors show brief spikes during the swing phase.
   - Detect heel-strike (signal rises past the threshold) and toe-off (signal falls below it).
   - Compute step and stride times.
   - Compare to GAITRite and return the error E (RMSE, ms).
   - No penalty terms: E is pure gait-timing error. Constraints are enforced by the optimizer (see step 6).
5. **Cache.** Save the preprocessed data to disk so later runs load in seconds.

## File 2: `optimize.py` — run the optimizers and generate outputs

6. **MOPSO (main method).** Each particle = N + 16 (x, y) pairs, using only the first N pairs. The swarm minimizes error and sensor count simultaneously. An external archive keeps the best error found for each N; that archive is the Pareto front.
   - **Hard constraints, no penalty.** Every active sensor must lie fully inside the foot (r ≤ x ≤ w − r, r ≤ y ≤ l − r) and no two active sensors may overlap or touch (center distance > 2r). These are never traded off against error. A layout that breaks either rule is never scored or stored.
   - **How:** start every particle at a feasible layout. After each move, clamp positions to the bounds. If active sensors then overlap, repair the layout by pushing overlapping pairs apart and re-clamping. If repair fails, the particle keeps its previous feasible position.
7. **PSO sweep (check).** For each N = 2…16, run a standard single-objective PSO on error. Its error-per-N curve should match the MOPSO front.
8. **SLSQP comparison.** Run SciPy's gradient-based solver from multiple random starts for N = 4 and N = 6 to show local-minimum trapping and non-convexity.
9. **Pick the optimum.** Apply f = W₁·E + W₂·N to the front across several weight choices to find the elbow point.
10. **Cross-cadence validation.** Take the optimal layouts (trained on NP only) and score them on Slow Pace (SP) and Fast Pace (FP) trials without re-optimizing.
11. **Export deliverables** (steps 6–11 run once per detection method, written to `results/sum/` and `results/first_last/`, plus `results/method_comparison.png` comparing the two):
    - `pareto_front.png`: error vs. sensor count
    - `optimal_layouts.png`: insole with pressure heatmap and optimal sensors for N = 3, 4, 5, 8
    - `convergence_comparison.png`: MOPSO/PSO vs. multi-start SLSQP
    - `cross_cadence_validation.csv`: NP-trained layouts tested on SP and FP
