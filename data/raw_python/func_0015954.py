def get_events(self, service_location_id, appliance_id, start, end,
                   max_number=None):
        """
        Request events for a given appliance

        Parameters
        ----------
        service_location_id : int
        appliance_id : int
        start : int | dt.datetime | pd.Timestamp
        end : int | dt.datetime | pd.Timestamp
            start and end support epoch (in milliseconds),
            datetime and Pandas Timestamp
            timezone-naive datetimes are assumed to be in UTC
        max_number : int, optional
            The maximum number of events that should be returned by this query
            Default returns all events in the selected period

        Returns
        -------
        dict
        """
        start = self._to_milliseconds(start)
        end = self._to_milliseconds(end)

        url = urljoin(URLS['servicelocation'], service_location_id, "events")
        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        params = {
            "from": start,
            "to": end,
            "applianceId": appliance_id,
            "maxNumber": max_number
        }
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()