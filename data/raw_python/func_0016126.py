async def get_band(self, tag):
        """Gets a band.

        Gets a band with specified tag. If no tag is specified, the request will fail.
        If the tag is invalid, a brawlstars.InvalidTag will be raised.
        If the data is missing, a ValueError will be raised.
        If the connection times out, a brawlstars.Timeout will be raised.
        If the data was unable to be received, a brawlstars.HTTPError will be raised along with the
        HTTP status code.
        On success, will return a Band.
        """

        tag = tag.strip("#")
        tag = tag.upper()

        try:
            async with self.session.get(self._base_url + 'bands/' + tag, timeout=self.timeout,
                                        headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                elif 500 > resp.status > 400:
                    raise HTTPError(resp.status)
                else:
                    raise Error()
        except asyncio.TimeoutError:
            raise Timeout()
        except ValueError:
            raise MissingData('data')
        except Exception:
            raise InvalidArg('tag')

        data = Box(data)
        band = Band(data)
        return band