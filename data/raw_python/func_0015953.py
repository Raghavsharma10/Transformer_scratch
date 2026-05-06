def _get_consumption(self, url, start, end, aggregation):
        """
        Request for both the get_consumption and
        get_sensor_consumption methods.

        Parameters
        ----------
        url : str
        start : dt.datetime
        end : dt.datetime
        aggregation : int

        Returns
        -------
        dict
        """
        start = self._to_milliseconds(start)
        end = self._to_milliseconds(end)

        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        params = {
            "aggregation": aggregation,
            "from": start,
            "to": end
        }
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()