async def _async_request_soup(url):
    '''
    Perform a GET web request and return a bs4 parser
    '''
    from bs4 import BeautifulSoup
    import aiohttp
    _LOGGER.debug('GET %s', url)
    async with aiohttp.ClientSession() as session:
        resp = await session.get(url)
        text = await resp.text()
        return BeautifulSoup(text, 'html.parser')