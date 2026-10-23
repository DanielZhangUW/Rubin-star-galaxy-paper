# Star-component parameters

One CSV per band combination (`<subset>.csv`), one row per r-magnitude bin.
Model: v10 with the paper-confirmed mean centering. The upstream implementation omits muU; this run subtracts its fitted per-band means before D construction.

The star (unresolved) density in `D = sqrt(sum_i ((dm_i - muU_i) / sigU_i)^2)` is

    f_star(D) = weight * D^a * exp(-b D^2) / norm(a, b) = C_unr * D^a * exp(-b D^2)
    norm(a, b) = Gamma((a + 1) / 2) / (2 b^((a + 1) / 2))

| column | meaning |
|---|---|
| `mag_lo`, `mag_hi`, `mag_center` | r-magnitude bin |
| `n_objects` | objects in the bin with a finite D for this subset |
| `star_source` | `fitted`: stage-1 fit to the pure core bump (D < 8, r < 24); `extrapolated`: weight from the log-linear trend, shape held at the plateau; `unfitted`: too few objects, not used |
| `weight` | star fraction W in the bin |
| `a`, `b` | star shape |
| `width_sigU` | 1 / sqrt(2 b), the star spread in units of the single-band sigU |
| `C_unr` | amplitude, weight / norm(a, b) |
| `n_core`, `core_fraction`, `core_mass` | stage-1 core fit (`fitted` rows only): objects with D < 8, their fraction of the bin, and the fraction of the fitted shape below D = 8 |
| `galaxy_fit_success` | whether the bin's galaxy fit converged; failed bins are not used in scoring |

For an object at magnitude m, `pS` uses `weight`, `a` and `b` interpolated
linearly between bin centres, held at the first centre brighter than it, and
past the last centre follows `star_model.json`: `log10 W = polyval(
log10_weight_poly, m)` with `a`, `b` at their plateau values.
