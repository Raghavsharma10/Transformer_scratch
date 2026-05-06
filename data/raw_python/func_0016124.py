def get_band(self, tag):
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
            resp = requests.get(self._base_url + 'bands/' + tag, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
            elif 500 > resp.status_code > 400:
                raise HTTPError(resp.status_code)
            else:
                raise Error()
        except ValueError:
            raise MissingData('data')
        except Exception:
            raise Timeout()

        data = Box(data)
        band = Band(data)
        return band