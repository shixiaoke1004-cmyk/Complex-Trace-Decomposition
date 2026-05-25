# Complex-Trace-Decomposition

A command-line tool for decomposing 4D seismic difference signals into orthogonal **amplitude** and **phase** components using the complex trace method.

Given a baseline and a monitor seismic volume, the tool computes the following identity exactly:

```
x2 - x1 = (A2 - A1) * C̄  +  Ā * (C2 - C1)
           └── amp component ─┘   └── phase component ─┘
```

where `A = |z|` is the instantaneous amplitude and `C = x / A = cos(φ)` is the instantaneous phase cosine of the analytic signal `z = x + iH[x]`.

---

## Features

- Exact amplitude / phase decomposition with closure residual diagnostics
- Memory-efficient chunk-wise processing via `numpy.memmap`
- Supports arbitrary N-dimensional volumes with configurable time axis
- Configurable input/output dtype, byte order, and array storage order (C / Fortran)
- Optional output of phase-matched and amplitude-matched volumes
- Optional output of instantaneous amplitude and phase cosine attributes
- JSON metadata with RMS and NRMS statistics per output volume

---

## Installation

```bash
pip install numpy
```

No additional dependencies are required.

---

## Usage

```bash
python complex_trace_4d.py \
    --base     baseline.bin  \
    --monitor  monitor.bin   \
    --shape    500,200,300   \
    --time-axis 0            \
    --dtype    float32       \
    --out-dir  results/      \
    --prefix   survey_A
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--base` | required | Baseline raw binary file |
| `--monitor` | required | Monitor raw binary file |
| `--shape` | required | Array shape in file order, e.g. `500,200,300` |
| `--time-axis` | `0` | Axis index corresponding to time samples |
| `--dtype` | `float32` | Input data type |
| `--output-dtype` | `float32` | Output data type |
| `--endian` | `native` | Byte order: `native`, `little`, or `big` |
| `--order` | `C` | Array storage order: `C` (row-major) or `F` (column-major) |
| `--chunk-traces` | `4096` | Number of traces processed per chunk |
| `--eps` | `1e-12` | Amplitude floor for stable division `C = x / A` |
| `--out-dir` | `complex_trace_outputs/` | Output directory |
| `--prefix` | `ct4d` | Filename prefix for all output files |
| `--write-matched` | off | Also write phase-matched and amplitude-matched volumes |
| `--write-attributes` | off | Also write instantaneous amplitude and phase cosine volumes |

---

## Outputs

All outputs are raw binary files with the same shape and storage order as the input.

| File | Description |
|---|---|
| `{prefix}_full_diff.bin` | `x2 - x1` |
| `{prefix}_amp_component.bin` | `(A2 - A1) * C̄` |
| `{prefix}_phase_component.bin` | `Ā * (C2 - C1)` |
| `{prefix}_closure.bin` | Residual: `full_diff - amp - phase` (should be near zero) |
| `{prefix}_metadata.json` | RMS, NRMS, and run configuration |

With `--write-matched`:

| File | Description |
|---|---|
| `{prefix}_base_phase_matched.bin` | `A1 * C̄` |
| `{prefix}_monitor_phase_matched.bin` | `A2 * C̄` |
| `{prefix}_base_amplitude_matched.bin` | `Ā * C1` |
| `{prefix}_monitor_amplitude_matched.bin` | `Ā * C2` |

With `--write-attributes`:

| File | Description |
|---|---|
| `{prefix}_base_inst_amp.bin` | Instantaneous amplitude of baseline `A1` |
| `{prefix}_monitor_inst_amp.bin` | Instantaneous amplitude of monitor `A2` |
| `{prefix}_base_phase_cos.bin` | Phase cosine of baseline `C1` |
| `{prefix}_monitor_phase_cos.bin` | Phase cosine of monitor `C2` |

---

## Metadata

A JSON file is written alongside the binary outputs:

```json
{
  "method": "complex trace 4D decomposition",
  "formula": "x2-x1 = (A2-A1)*(C1+C2)/2 + (A1+A2)*(C2-C1)/2",
  "rms": {
    "base": 0.412,
    "monitor": 0.419,
    "full_diff": 0.053,
    "amp_component": 0.038,
    "phase_component": 0.032,
    "closure": 0.000
  },
  "nrms": {
    "full_diff_percent": 12.7,
    "amp_component_percent": 9.1,
    "phase_component_percent": 7.7
  },
  "closure_rms_ratio_to_full_diff": 0.0000021,
  "closure_rms_ratio_percent": 0.00021
}
```

NRMS is defined following Johnston (2013):

$$\text{NRMS}\\% = \frac{200 \times \text{RMS}(\Delta)}{\text{RMS}(x_1) + \text{RMS}(x_2)}$$

---

## References

- Taner, M. T., Koehler, F., & Sheriff, R. E. (1979). Complex seismic trace analysis. *Geophysics*, 44(6), 1041–1063.
- Johnston, D. H. (2013). *Methods and Applications in Reservoir Geophysics*. SEG.

