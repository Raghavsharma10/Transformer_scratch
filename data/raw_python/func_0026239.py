async def get_stats(self, battletag: str, regions=(EUROPE, KOREA, AMERICAS, CHINA, JAPAN, ANY),
                        platform=None, _session=None, handle_ratelimit=None, max_tries=None, request_timeout=None):
        """Returns the stats for the profiles on the specified regions and platform. The format for regions without a matching user, the format is the same as get_profile.
        The stats are returned in a dictionary with a similar format to what https://github.com/SunDwarf/OWAPI/blob/master/api.md#get-apiv3ubattletagstats specifies."""

        if platform is None:
            platform = self.default_platform
        try:
            blob_dict = await self._base_request(battletag, "stats", _session, platform=platform,
                                                 handle_ratelimit=handle_ratelimit, max_tries=max_tries,
                                                 request_timeout=request_timeout)
        except ProfileNotFoundError as e:
            # The battletag doesn't exist
            blob_dict = {}
        existing_regions = {key: val for key, val in blob_dict.items() if ((val is not None) and (key != "_request"))}
        return {key: [inner_val for inner_key, inner_val in val.items() if inner_key == "stats"][0] for key, val in
                existing_regions.items() if key in regions}