from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not shape or any(n <= 0 for n in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def build_dtype(name: str, endian: str) -> np.dtype:
    dtype = np.dtype(name)
    if dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"unsupported dtype {name!r}; use a numeric dtype")
    if dtype.itemsize == 1 or endian == "native":
        return dtype
    byteorder = "<" if endian == "little" else ">"
    return dtype.newbyteorder(byteorder)


def analytic_signal_fft(traces: np.ndarray) -> np.ndarray:
    n_samples = traces.shape[-1]
    spectrum = np.fft.fft(traces, axis=-1)
    multiplier = np.zeros(n_samples, dtype=np.float64)

    if n_samples % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n_samples // 2] = 1.0
        multiplier[1 : n_samples // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n_samples + 1) // 2] = 2.0

    return np.fft.ifft(spectrum * multiplier, axis=-1)


def trace_chunks(
    base: np.ndarray,
    monitor: np.ndarray,
    time_axis: int,
    chunk_traces: int,
) -> Iterable[tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]]:
    base_view = np.moveaxis(base, time_axis, -1)
    monitor_view = np.moveaxis(monitor, time_axis, -1)
    spatial_shape = base_view.shape[:-1]
    n_traces = int(np.prod(spatial_shape, dtype=np.int64))

    for start in range(0, n_traces, chunk_traces):
        stop = min(start + chunk_traces, n_traces)
        flat = np.arange(start, stop)
        coords = np.unravel_index(flat, spatial_shape)
        index = coords + (slice(None),)
        yield index, np.asarray(base_view[index]), np.asarray(monitor_view[index])


def rms_accumulate(values: np.ndarray) -> tuple[float, int]:
    values64 = np.asarray(values, dtype=np.float64)
    return float(np.sum(values64 * values64)), int(values64.size)


def write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")


def decompose(args: argparse.Namespace) -> dict:
    shape = args.shape
    ndim = len(shape)

    if not (-ndim <= args.time_axis < ndim):
        raise ValueError(
            f"--time-axis {args.time_axis} is out of range for a {ndim}-dimensional array"
        )
    time_axis = args.time_axis % ndim

    input_dtype = build_dtype(args.dtype, args.endian)
    output_dtype = build_dtype(args.output_dtype, args.endian)
    expected_count = int(np.prod(shape, dtype=np.int64))
    expected_bytes = expected_count * input_dtype.itemsize

    for label, path in (("base", args.base), ("monitor", args.monitor)):
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{label} file size mismatch: expected {expected_bytes} bytes "
                f"from --shape/--dtype, got {actual_bytes} bytes"
            )

    if args.chunk_traces <= 0:
        raise ValueError("--chunk-traces must be positive")

    base = np.memmap(args.base, dtype=input_dtype, mode="r", shape=shape, order=args.order)
    monitor = np.memmap(args.monitor, dtype=input_dtype, mode="r", shape=shape, order=args.order)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "full_diff": np.memmap(
            args.out_dir / f"{args.prefix}_full_diff.bin",
            dtype=output_dtype, mode="w+", shape=shape, order=args.order,
        ),
        "amp_component": np.memmap(
            args.out_dir / f"{args.prefix}_amp_component.bin",
            dtype=output_dtype, mode="w+", shape=shape, order=args.order,
        ),
        "phase_component": np.memmap(
            args.out_dir / f"{args.prefix}_phase_component.bin",
            dtype=output_dtype, mode="w+", shape=shape, order=args.order,
        ),
        "closure": np.memmap(
            args.out_dir / f"{args.prefix}_closure.bin",
            dtype=output_dtype, mode="w+", shape=shape, order=args.order,
        ),
    }

    if args.write_matched:
        for name in (
            "base_phase_matched",
            "monitor_phase_matched",
            "base_amplitude_matched",
            "monitor_amplitude_matched",
        ):
            outputs[name] = np.memmap(
                args.out_dir / f"{args.prefix}_{name}.bin",
                dtype=output_dtype, mode="w+", shape=shape, order=args.order,
            )

    if args.write_attributes:
        for name in ("base_inst_amp", "monitor_inst_amp", "base_phase_cos", "monitor_phase_cos"):
            outputs[name] = np.memmap(
                args.out_dir / f"{args.prefix}_{name}.bin",
                dtype=output_dtype, mode="w+", shape=shape, order=args.order,
            )

    output_views = {name: np.moveaxis(array, time_axis, -1) for name, array in outputs.items()}

    primary_keys = ("full_diff_sq", "amp_component_sq", "phase_component_sq", "closure_sq")
    sums: dict[str, float] = {
        "base_sq": 0.0,
        "monitor_sq": 0.0,
        **{k: 0.0 for k in primary_keys},
    }
    sample_count = 0

    for index, base_chunk, monitor_chunk in trace_chunks(base, monitor, time_axis, args.chunk_traces):
        x1 = base_chunk.astype(np.float64, copy=False)
        x2 = monitor_chunk.astype(np.float64, copy=False)

        z1 = analytic_signal_fft(x1)
        z2 = analytic_signal_fft(x2)
        a1 = np.abs(z1)
        a2 = np.abs(z2)

        c1 = np.divide(x1, a1, out=np.zeros_like(x1), where=a1 > args.eps)
        c2 = np.divide(x2, a2, out=np.zeros_like(x2), where=a2 > args.eps)

        abar = 0.5 * (a1 + a2)
        cbar = 0.5 * (c1 + c2)
        full_diff = x2 - x1
        amp_component = (a2 - a1) * cbar
        phase_component = abar * (c2 - c1)
        closure = full_diff - amp_component - phase_component

        output_views["full_diff"][index] = full_diff.astype(output_dtype, copy=False)
        output_views["amp_component"][index] = amp_component.astype(output_dtype, copy=False)
        output_views["phase_component"][index] = phase_component.astype(output_dtype, copy=False)
        output_views["closure"][index] = closure.astype(output_dtype, copy=False)

        if args.write_matched:
            output_views["base_phase_matched"][index] = (a1 * cbar).astype(output_dtype, copy=False)
            output_views["monitor_phase_matched"][index] = (a2 * cbar).astype(output_dtype, copy=False)
            output_views["base_amplitude_matched"][index] = (abar * c1).astype(output_dtype, copy=False)
            output_views["monitor_amplitude_matched"][index] = (abar * c2).astype(output_dtype, copy=False)

        if args.write_attributes:
            output_views["base_inst_amp"][index] = a1.astype(output_dtype, copy=False)
            output_views["monitor_inst_amp"][index] = a2.astype(output_dtype, copy=False)
            output_views["base_phase_cos"][index] = c1.astype(output_dtype, copy=False)
            output_views["monitor_phase_cos"][index] = c2.astype(output_dtype, copy=False)

        chunk_pairs = (
            ("base_sq", x1),
            ("monitor_sq", x2),
            ("full_diff_sq", full_diff),
            ("amp_component_sq", amp_component),
            ("phase_component_sq", phase_component),
            ("closure_sq", closure),
        )
        chunk_count = None
        for key, values in chunk_pairs:
            add_sum, add_count = rms_accumulate(values)
            sums[key] += add_sum
            if chunk_count is None:
                chunk_count = add_count
        sample_count += chunk_count

    for array in outputs.values():
        array.flush()

    rms = {
        key.removesuffix("_sq"): float(np.sqrt(value / sample_count)) if sample_count else 0.0
        for key, value in sums.items()
    }
    nrms_denominator = rms["base"] + rms["monitor"]

    def _nrms(key: str) -> float | None:
        return 200.0 * rms[key] / nrms_denominator if nrms_denominator else None

    nrms = {
        "full_diff_percent": _nrms("full_diff"),
        "amp_component_percent": _nrms("amp_component"),
        "phase_component_percent": _nrms("phase_component"),
    }

    closure_ratio = (
        rms["closure"] / rms["full_diff"] if rms["full_diff"] else None
    )

    metadata = {
        "method": "complex trace 4D decomposition",
        "formula": "x2-x1 = (A2-A1)*(C1+C2)/2 + (A1+A2)*(C2-C1)/2",
        "base": str(args.base),
        "monitor": str(args.monitor),
        "shape": shape,
        "time_axis": time_axis,
        "input_dtype": str(input_dtype),
        "output_dtype": str(output_dtype),
        "order": args.order,
        "chunk_traces": args.chunk_traces,
        "eps": args.eps,
        "outputs": {name: str(args.out_dir / f"{args.prefix}_{name}.bin") for name in outputs},
        "rms": rms,
        "nrms": nrms,
        "closure_rms_ratio_to_full_diff": closure_ratio,
        "closure_rms_ratio_percent": (
            100.0 * closure_ratio if closure_ratio is not None else None
        ),
    }
    write_metadata(args.out_dir / f"{args.prefix}_metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose two 4D seismic binary volumes into amplitude and phase components."
    )
    parser.add_argument("--base", type=Path, required=True, help="baseline raw .bin file")
    parser.add_argument("--monitor", type=Path, required=True, help="monitor raw .bin file")
    parser.add_argument(
        "--shape",
        type=parse_shape,
        required=True,
        help="array shape in file order, for example 500,200,300",
    )
    parser.add_argument("--time-axis", type=int, default=0, help="axis index of time samples")
    parser.add_argument("--dtype", default="float32", help="input dtype, default float32")
    parser.add_argument("--output-dtype", default="float32", help="output dtype, default float32")
    parser.add_argument(
        "--endian",
        choices=("native", "little", "big"),
        default="native",
        help="binary byte order",
    )
    parser.add_argument("--order", choices=("C", "F"), default="C", help="array storage order")
    parser.add_argument("--out-dir", type=Path, default=Path("complex_trace_outputs"))
    parser.add_argument("--prefix", default="ct4d", help="output filename prefix")
    parser.add_argument("--chunk-traces", type=int, default=4096, help="number of traces per chunk")
    parser.add_argument("--eps", type=float, default=1.0e-12, help="amplitude floor for C=x/A")
    parser.add_argument("--write-matched", action="store_true", help="also output matched volumes")
    parser.add_argument("--write-attributes", action="store_true", help="also output A and cos(phase)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = decompose(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
