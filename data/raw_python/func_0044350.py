async def one(s: Sublemon):
    """Spin up some subprocesses, sleep, and echo a message for this coro."""
    shell_cmds = [
        'sleep 1 && echo subprocess 1 in coroutine one',
        'sleep 1 && echo subprocess 2 in coroutine one']
    async for line in s.iter_lines(*shell_cmds):
        print(line)