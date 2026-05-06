async def open_pipe_connection(
    path=None,
    *,
    loop=None,
    limit=DEFAULT_LIMIT,
    **kwargs
):
    """
    Connect to a server using a Windows named pipe.
    """
    path = path.replace('/', '\\')
    loop = loop or asyncio.get_event_loop()

    reader = asyncio.StreamReader(limit=limit, loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    transport, _ = await loop.create_pipe_connection(
        lambda: protocol,
        path,
        **kwargs
    )
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)

    return reader, writer