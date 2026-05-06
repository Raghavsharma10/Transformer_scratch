async def post(self, url_path: str, params: dict = None, rtype: str = RESPONSE_JSON, schema: dict = None) -> Any:
        """
        POST request on self.endpoint + url_path

        :param url_path: Url encoded path following the endpoint
        :param params: Url query string parameters dictionary
        :param rtype: Response type
        :param schema: Json Schema to validate response (optional, default None)
        :return:
        """
        if params is None:
            params = dict()

        client = API(self.endpoint.conn_handler(self.session, self.proxy))

        # get aiohttp response
        response = await client.requests_post(url_path, **params)

        # if schema supplied...
        if schema is not None:
            # validate response
            await parse_response(response, schema)

        # return the chosen type
        if rtype == RESPONSE_AIOHTTP:
            return response
        elif rtype == RESPONSE_TEXT:
            return await response.text()
        elif rtype == RESPONSE_JSON:
            return await response.json()