async def amerge(*agens) -> AsyncGenerator[Any, None]:
    """Thin wrapper around aiostream.stream.merge."""
    xs = stream.merge(*agens)
    async with xs.stream() as streamer:
        async for x in streamer:
            yield x