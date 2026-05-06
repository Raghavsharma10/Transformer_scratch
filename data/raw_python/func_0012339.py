async def async_set_summary(program):
    '''
    Set a program's summary
    '''
    import aiohttp
    async with aiohttp.ClientSession() as session:
        resp = await session.get(program.get('url'))
        text = await resp.text()
        summary = extract_program_summary(text)
        program['summary'] = summary
        return program