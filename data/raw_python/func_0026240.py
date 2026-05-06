async def _base_request(self, battle_tag: str, endpoint_name: str, session: aiohttp.ClientSession, *, platform=None,
                            handle_ratelimit=None, max_tries=None, request_timeout=None):
        """Does a request to some endpoint. This is also where ratelimit logic is handled."""
        # We check the different optional arguments, and if they're not passed (are none) we set them to the default for the client object
        if platform is None:
            platform = self.default_platform
        if handle_ratelimit is None:
            handle_ratelimit = self.default_handle_ratelimit
        if max_tries is None:
            max_tries = self.default_max_tries
        if request_timeout is None:
            request_timeout = self.default_request_timeout

        # The battletag with #s removed
        san_battle_tag = self.sanitize_battletag(battle_tag)

        # The ratelimit logic
        for _ in range(max_tries):
            # We execute a request
            try:
                resp_json, status = await self._async_get(
                    session,
                    self.server_url + self._api_urlpath + "{battle_tag}/{endpoint}".format(
                        battle_tag=san_battle_tag,
                        endpoint=endpoint_name
                    ),
                    params={"platform": platform},
                    # Passed to _async_get and indicates what platform we're searching on
                    headers={"User-Agent": "overwatch_python_api"},
                    # According to https://github.com/SunDwarf/OWAPI/blob/master/owapi/v3/v3_util.py#L18 we have to customise our User-Agent, so we do
                    _async_timeout_seconds=request_timeout
                )
                if status == 429 and resp_json["msg"] == "you are being ratelimited":
                    raise RatelimitError
            except RatelimitError as e:
                # This excepts both RatelimitErrors and TimeoutErrors, ratelimiterrors for server returning a ratelimit, timeouterrors for the connection not being done in with in the timeout
                # We are ratelimited, so we check if we handle ratelimiting logic
                # If so, we wait and then execute the next iteration of the loop
                if handle_ratelimit:
                    # We wait to remedy ratelimiting, and we wait a bit more than the response says we should
                    await asyncio.sleep(resp_json["retry"] + 1)
                    continue
                else:
                    raise
            else:
                # We didn't get an error, so we exit the loop because it was a successful request
                break
        else:
            # The loop didn't stop because it got breaked, which means that we got ratelimited until the maximum number of tries were finished
            raise RatelimitError("Got ratelimited for each requests until the maximum number of retries were reached.")

        # Validate the response
        if status != 200:
            if status == 404 and resp_json["msg"] == "profile not found":
                raise ProfileNotFoundError(
                    "Got HTTP 404, profile not found. This is caused by the given battletag not existing on the specified platform.")
            if status == 429 and resp_json["msg"] == "you are being ratelimited":
                raise RatelimitError(
                    "Got HTTP 429, you are being ratelimited. This is caused by calls to the api too frequently.")
            raise ConnectionError("Did not get HTTP status 200, got: {0}".format(status))
        return resp_json