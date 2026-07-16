# Flood request simulation: Locust results

## How the test was run

I flooded the API with Locust, posting a real 64×64 chest X-ray to `POST /predict` (8 parts
prediction to 1 part `GET /status`, matching the task weights in `locust/locustfile.py`).

| Setting | Value |
|---|---|
| Tool | Locust 2.45.0 (headless) |
| Concurrent users | 50 |
| Spawn rate | 25 users/second |
| Duration | ~15 s per run |
| Payload | one real chest X-ray PNG per request |
| Test host | 4 vCPU / 3.9 GB RAM Linux container |
| Command | `locust -f locust/locustfile.py --headless -u 50 -r 25 -t 15s --host http://localhost:8000 --csv results/w1` |

How the replicas were scaled: Docker was not available in the environment where I took these
measurements, so each replica is a uvicorn worker process (`--workers N`) rather than a separate
Docker container. Each worker is an independent OS process with its own TensorFlow runtime and its
own copy of the model, competing for the same CPU and memory that separate containers on one host
would, so the scaling trend is representative. To reproduce with genuine containers:

```bash
docker compose up --build --scale api=1    # then 2, then 4
locust -f locust/locustfile.py --headless -u 50 -r 25 -t 15s --host http://localhost:8080 --csv results/scale_1
```

## Results

| Replicas | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Max (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 worker  | 529 | 0 (0.00%) | 134 | 90 | 490 | 748 | 35.71 |
| 2 workers | 544 | 0 (0.00%) | 62  | 36 | 280 | 353 | 39.21 |
| 4 workers | 511 | 0 (0.00%) | 53  | 30 | 250 | 367 | 39.66 |
| 4 workers (repeat) | 492 | 0 (0.00%) | 43 | 28 | 250 | 329 | 38.41 |

Raw Locust CSV exports: `locust_1_worker_stats.csv`, `locust_2_workers_stats.csv`,
`locust_4_workers_stats.csv`.

## How the model responded to the flood

No requests failed at any scale. Under a 50-user flood the service never dropped a request, never
returned a 5xx, and never crashed. It slowed down, but only in latency. The model is cheap (a small
CNN on a 64×64 grayscale image), so the per-request cost is dominated by request handling and the
TensorFlow call rather than the network.

Throughput is capped by the load generator, not the server, so read latency instead of req/s. This
is the most important caveat in the whole test. With 50 users and `wait_time = between(0.5, 2.0)`
(mean 1.25 s), the theoretical ceiling is about `50 / 1.25 ≈ 40 req/s`. Every configuration with 2 or
more workers lands at 38 to 40 req/s, which means the load generator ran out of requests to send
before the server ran out of capacity. Those throughput numbers measure Locust's offered load, not
the API's limit. Concluding "the server maxes out at 40 req/s" from this data would be wrong. Latency
is the signal that still discriminates.

Going from 1 to 2 replicas is a clear, real win. Median latency dropped from 90 ms to 36 ms (2.5×
faster) and p95 from 490 ms to 280 ms. With a single worker, requests queue behind one process. The
second worker absorbs the queue and the tail shrinks sharply.

Going from 2 to 4 replicas gives diminishing returns. The median improved only from 36 ms to 30 ms,
and p95 from 280 ms to 250 ms. With 4 vCPUs and TensorFlow already running multi-threaded inside each
worker, the fourth replica has little headroom left. The curve is flattening, which is what you
expect as the replica count approaches the physical core count.

## A correction worth recording

An earlier run of this same 4-worker configuration (against the previous model) produced a much worse
result: 25.19 req/s with a 420 ms median and a 3,110 ms max, worse than a single worker. I first wrote
it up as evidence that more containers is not automatically faster.

It did not reproduce. Re-running 4 workers twice gave medians of 30 ms and 28 ms with no degradation.
The original figure was almost certainly an artefact of memory pressure and CPU contention on a
shared box right after a training run, not a property of the architecture.

The honest conclusion is the less dramatic one: a single 15-second run on a shared 4-vCPU host is
noisy, and one measurement is not a result. I kept the anomaly here instead of deleting it, because
silently swapping in whichever number tells the better story is exactly the failure mode load testing
is supposed to prevent. Trustworthy numbers would need repeated runs, a quiet host, and reported
variance.

## Conclusion

1. The service is correct under load, with 0 failures across roughly 2,000 requests at every scale.
2. Scaling from 1 to 2 replicas is clearly worth it (2.5× lower median latency). Going from 2 to 4
   buys little on a 4-core host.
3. This test cannot find the server's ceiling, because the load generator saturates first. Measuring
   real capacity would need `wait_time = constant(0)` or several hundred users.
4. Constrain intra-process threading (`OMP_NUM_THREADS=1`, set in `render.yaml`) before adding
   replicas, so each worker stays single-threaded and replicas scale cleanly instead of fighting over
   cores.
5. Past a host's core count, scale across more hosts. That is what a cloud autoscaler does, and why
   the deployment is containerised.
