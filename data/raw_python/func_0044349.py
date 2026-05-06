async def main():
    """`sublemon` library example!"""
    for c in (1, 2, 4,):
        async with Sublemon(max_concurrency=c) as s:
            start = time.perf_counter()
            await asyncio.gather(one(s), two(s))
            end = time.perf_counter()
            print('Limiting to', c, 'concurrent subprocess(es) took',
                  end-start, 'seconds\n')