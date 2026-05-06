async def two(s: Sublemon):
    """Spin up some subprocesses, sleep, and echo a message for this coro."""
    subprocess_1, subprocess_2 = s.spawn(
        'sleep 1 && echo subprocess 1 in coroutine two',
        'sleep 1 && echo subprocess 2 in coroutine two')
    async for line in amerge(subprocess_1.stdout, subprocess_2.stdout):
        print(line.decode('utf-8'), end='')