async def write_request(
        self, method: constants.HttpRequestMethod, *,
        uri: str="/", authority: Optional[str]=None,
        scheme: Optional[str]=None,
        headers: Optional[_HeaderType]=None) -> \
            "writers.HttpRequestWriter":
        """
        Send next request to the server.
        """
        return await self._delegate.write_request(
            method, uri=uri, authority=authority,
            scheme=scheme, headers=headers)