def get_sensor_consumption(self, service_location_id, sensor_id, start,
                               end, aggregation):
        """
        Request consumption for a given sensor in a given service location

        Parameters
        ----------
        service_location_id : int
        sensor_id : int
        start : int | dt.datetime | pd.Timestamp
        end : int | dt.datetime | pd.Timestamp
            start and end support epoch (in milliseconds),
            datetime and Pandas Timestamp
            timezone-naive datetimes are assumed to be in UTC
        aggregation : int
            1 = 5 min values (only available for the last 14 days)
            2 = hourly values
            3 = daily values
            4 = monthly values
            5 = quarterly values

        Returns
        -------
        dict
        """
        url = urljoin(URLS['servicelocation'], service_location_id, "sensor",
                      sensor_id, "consumption")
        return self._get_consumption(url=url, start=start, end=end,
                                     aggregation=aggregation)