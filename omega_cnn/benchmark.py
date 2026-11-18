#!/usr/bin/env python3
"""Compare ONNX outputs, accuracy (when labels exist), and latency."""
import argparse
import json
import time
import numpy as np
import onnxruntime as ort


def benchmark(model_path, samples, labels=None, runs=30):
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name
    outputs = []
    times = []
    for sample in samples:
        feed = {name: sample[np.newaxis].astype(np.float32)}
        session.run(None, feed)  # warm-up
        start = time.perf_counter()
        for _ in range(runs):
            output = session.run(None, feed)[0]
        times.append((time.perf_counter() - start) * 1000 / runs)
        outputs.append(output[0])
    scores = np.asarray(outputs)
    result = {'latency_mean_ms': float(np.mean(times)), 'latency_p95_ms': float(np.percentile(times, 95)), 'predictions': scores.argmax(axis=-1).tolist()}
    if labels is not None:
        result['accuracy'] = float(np.mean(scores.argmax(axis=-1) == labels))
    return result, scores

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('candidate')
    p.add_argument('dataset', help='NPZ containing x, optionally y; input in native model shape')
    p.add_argument('--runs', type=int, default=30)
    p.add_argument('--max-absolute-error', type=float, default=0.02)
    p.add_argument('--max-disagreement', type=float, default=0.0)
    a = p.parse_args()
    if a.runs < 1: p.error('--runs must be positive')
    with np.load(a.dataset, allow_pickle=False) as data:
        x = data['x']
        y = data['y'] if 'y' in data.files else None
    if len(x) == 0: p.error('dataset must contain at least one sample')
    original, ref = benchmark(a.baseline, x, y, a.runs)
    candidate, pred = benchmark(a.candidate, x, y, a.runs)
    if ref.shape != pred.shape: raise SystemExit('Output shape mismatch')
    max_error = float(np.max(np.abs(ref - pred)))
    disagreement = float(np.mean(ref.argmax(axis=-1) != pred.argmax(axis=-1)))
    report = {'baseline': original, 'candidate': candidate, 'max_absolute_error': max_error, 'prediction_disagreement': disagreement, 'passed': max_error <= a.max_absolute_error and disagreement <= a.max_disagreement}
    print(json.dumps(report, indent=2))
    if not report['passed']: raise SystemExit(2)
