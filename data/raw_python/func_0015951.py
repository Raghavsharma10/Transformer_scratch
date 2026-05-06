def get_consumption(self, service_location_id, start, end, aggregation, raw=False):
        """
        Request Elektricity consumption and Solar production
        for a given service location.

        Parameters
        ----------
        service_location_id : int
        start : int | dt.datetime | pd.Timestamp
        end : int | dt.datetime | pd.Timestamp
            start and end support epoch (in milliseconds),
            datetime and Pandas Timestamp
        aggregation : int
            1 = 5 min values (only available for the last 14 days)
            2 = hourly values
            3 = daily values
            4 = monthly values
            5 = quarterly values
        raw : bool
            default False
            if True: Return the data "as is" from the server
            if False: convert the 'alwaysOn' value to Wh.
            (the server returns this value as the sum of the power,
            measured in 5 minute blocks. This means that it is 12 times
            higher than the consumption in Wh.
            See https://github.com/EnergieID/smappy/issues/24)

        Returns
        -------
        dict
        """
        url = urljoin(URLS['servicelocation'], service_location_id,
                      "consumption")
        d = self._get_consumption(url=url, start=start, end=end,
                                  aggregation=aggregation)
        if not raw:
            for block in d['consumptions']:
                if 'alwaysOn' not in block.keys():
                    break
                block.update({'alwaysOn': block['alwaysOn'] / 12})
        return d