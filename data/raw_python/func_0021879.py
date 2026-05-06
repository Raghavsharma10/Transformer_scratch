def query(  # type: ignore
        self,
        url: Union[str, methods],
        data: Optional[MutableMapping] = None,
        headers: Optional[MutableMapping] = None,
        as_json: Optional[bool] = None,
    ) -> dict:
        """
        Query the slack API

        When using :class:`slack.methods` the request is made `as_json` if available

        Args:
            url: :class:`slack.methods` or url string
            data: JSON encodable MutableMapping
            headers: Custom headers
            as_json: Post JSON to the slack API
        Returns:
            dictionary of slack API response data

        """
        url, body, headers = sansio.prepare_request(
            url=url,
            data=data,
            headers=headers,
            global_headers=self._headers,
            token=self._token,
        )
        return self._make_query(url, body, headers)