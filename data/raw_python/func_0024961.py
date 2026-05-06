def create_timeseries(self, **kwargs):
        """
        Creates an instance of the Time Series Service.
        """
        ts = predix.admin.timeseries.TimeSeries(**kwargs)
        ts.create()

        client_id = self.get_client_id()
        if client_id:
            ts.grant_client(client_id)

        ts.add_to_manifest(self)
        return ts