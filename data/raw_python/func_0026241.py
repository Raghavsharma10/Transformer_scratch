async def _async_get(self, session: aiohttp.ClientSession, *args, _async_timeout_seconds: int = 5,
                         **kwargs):
        """Uses aiohttp to make a get request asynchronously. 
        Will raise asyncio.TimeoutError if the request could not be completed 
        within _async_timeout_seconds (default 5) seconds."""

        # Taken almost directly from the aiohttp tutorial
        with async_timeout.timeout(_async_timeout_seconds):
            async with session.get(*args, **kwargs) as response:
                return await response.json(), response.status