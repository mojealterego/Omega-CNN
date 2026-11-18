#!/usr/bin/env python3
"""Safe ONNX optimization pipeline; never replaces original weights."""
import argparse
import json
from pathlib import Path


def run(source: Path, destination: Path, quantize: bool) -> dict:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit('Install dependencies: pip install -r requirements.txt') from exc
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    nodes_before = len(model.graph.node)
    optimized = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(optimized)
    destination.mkdir(parents=True, exist_ok=True)
    inferred_path = destination / 'model_inferred.onnx'
    onnx.save(optimized, str(inferred_path))
    report = {'source': str(source), 'nodes_before': nodes_before, 'nodes_after_inference': len(optimized.graph.node), 'shape_inferred': str(inferred_path), 'quantization_requested': quantize}
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        optimized_path = destination / 'model_ort_optimized.onnx'
        session_options.optimized_model_filepath = str(optimized_path)
        ort.InferenceSession(str(inferred_path), sess_options=session_options, providers=['CPUExecutionProvider'])
        report['ort_optimized'] = str(optimized_path)
        if quantize:
            # Dynamic quantization is applied only to MatMul; Conv requires calibrated static quantization.
            quantized_path = destination / 'model_matmul_int8.onnx'
            quantize_dynamic(str(inferred_path), str(quantized_path), op_types_to_quantize=['MatMul'], weight_type=QuantType.QInt8)
            onnx.checker.check_model(onnx.load(str(quantized_path)))
            ort.InferenceSession(str(quantized_path), providers=['CPUExecutionProvider'])
            report['quantized'] = str(quantized_path)
    except ImportError:
        report['runtime_status'] = 'onnxruntime unavailable: shape inference only'
    (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.quantize), indent=2))
