async def iter(
        self,
        url: Union[str, methods],
        data: Optional[MutableMapping] = None,
        headers: Optional[MutableMapping] = None,
        *,
        limit: int = 200,
        iterkey: Optional[str] = None,
        itermode: Optional[str] = None,
        minimum_time: Optional[int] = None,
        as_json: Optional[bool] = None
    ) -> AsyncIterator[dict]:
        """
        Iterate over a slack API method supporting pagination

        When using :class:`slack.methods` the request is made `as_json` if available

        Args:
            url: :class:`slack.methods` or url string
            data: JSON encodable MutableMapping
            headers:
            limit: Maximum number of results to return per call.
            iterkey: Key in response data to iterate over (required for url string).
            itermode: Iteration mode (required for url string) (one of `cursor`, `page` or `timeline`)
            minimum_time: Minimum elapsed time (in seconds) between two calls to the Slack API (default to 0).
             If not reached the client will sleep for the remaining time.
            as_json: Post JSON to the slack API
        Returns:
            Async iterator over `response_data[key]`

        """
        itervalue = None

        if not data:
            data = {}

        last_request_time = None
        while True:
            current_time = time.time()
            if (
                minimum_time
                and last_request_time
                and last_request_time + minimum_time > current_time
            ):
                await self.sleep(last_request_time + minimum_time - current_time)

            data, iterkey, itermode = sansio.prepare_iter_request(
                url,
                data,
                iterkey=iterkey,
                itermode=itermode,
                limit=limit,
                itervalue=itervalue,
            )
            last_request_time = time.time()
            response_data = await self.query(url, data, headers, as_json)
            itervalue = sansio.decode_iter_request(response_data)
            for item in response_data[iterkey]:
                yield item

            if not itervalue:
                break