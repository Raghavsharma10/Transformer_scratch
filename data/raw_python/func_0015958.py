def get_consumption_dataframe(self, service_location_id, start, end,
                                  aggregation, sensor_id=None, localize=False,
                                  raw=False):
        """
        Extends get_consumption() AND get_sensor_consumption(),
        parses the results in a Pandas DataFrame

        Parameters
        ----------
        service_location_id : int
        start : dt.datetime | int
        end : dt.datetime | int
            timezone-naive datetimes are assumed to be in UTC
            epoch timestamps need to be in milliseconds
        aggregation : int
        sensor_id : int, optional
            If a sensor id is passed, api method get_sensor_consumption will
            be used otherwise (by default),
            the get_consumption method will be used: this returns Electricity
            and Solar consumption and production.
        localize : bool
            default False
            default returns timestamps in UTC
            if True, timezone is fetched from service location info and
            Data Frame is localized
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
        pd.DataFrame
        """
        import pandas as pd

        if sensor_id is None:
            data = self.get_consumption(
                service_location_id=service_location_id, start=start,
                end=end, aggregation=aggregation, raw=raw)
            consumptions = data['consumptions']
        else:
            data = self.get_sensor_consumption(
                service_location_id=service_location_id, sensor_id=sensor_id,
                start=start, end=end, aggregation=aggregation)
            # yeah please someone explain me why they had to name this
            # differently...
            consumptions = data['records']

        df = pd.DataFrame.from_dict(consumptions)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, unit='ms', utc=True)
            if localize:
                info = self.get_service_location_info(
                    service_location_id=service_location_id)
                timezone = info['timezone']
                df = df.tz_convert(timezone)
        return df