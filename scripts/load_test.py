import asyncio
import time
from pathlib import Path
import httpx
import numpy as np

IMAGE_PATH = Path(r"..\dataset\test\images\000000986v_jpg.rf.2060772a09d5eaf2a777b9385316749d.jpg")

def percentile(values, p):
    if not values:
        return float("nan")
    xs = sorted(values)
    k = int(np.ceil((p / 100.0) * len(xs)) - 1)
    k = max(0, min(len(xs) - 1, k))
    return xs[k]

async def one_request(client: httpx.AsyncClient, api_url: str, image_bytes: bytes):
    t0 = time.perf_counter()
    try:
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        response = await client.post(api_url, files=files, timeout=60.0)
        ok = (response.status_code == 200)
        return (time.perf_counter() - t0) * 1000.0, ok
    except Exception:
        return (time.perf_counter() - t0) * 1000.0, False

async def run_load(api_url: str, concurrency: int, total_requests: int):
    image_bytes = IMAGE_PATH.read_bytes()

    latencies = []
    ok_count = 0
    err_count = 0

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded():
            async with sem:
                return await one_request(client, api_url, image_bytes)

        tasks = [asyncio.create_task(bounded()) for _ in range(total_requests)]
        for t in asyncio.as_completed(tasks):
            ms, ok = await t
            latencies.append(ms)
            if ok:
                ok_count += 1
            else:
                err_count += 1

    duration = time.perf_counter() - start
    rps = total_requests / duration if duration > 0 else 0.0

    avg = sum(latencies) / len(latencies) if latencies else float("nan")
    p95 = percentile(latencies, 95.0)

    print(f"\n  Concurrency = {concurrency}, Requests = {total_requests}")
    print(f"  ok = {ok_count} err = {err_count}")
    print(f"  avg = {avg:.1f}ms p95 = {p95:.1f}ms")
    print(f"  throughput = {rps:.2f} req/s")

async def run_suite(label: str, api_url: str):
    print(f"\n=== {label} ===")
    for c in [1, 5, 10, 25]:
        await run_load(api_url=api_url, concurrency=c, total_requests=50)

async def main():
    await run_suite("annotate=true", "http://localhost:8000/detect?annotate=true")
    await run_suite("annotate=false", "http://localhost:8000/detect?annotate=false")

if __name__ == "__main__":
    asyncio.run(main())