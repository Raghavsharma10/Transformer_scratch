def create(self, asset, timeseries, client_id, client_secret,
            ui_client_id=None, ui_client_secret=None):
        """
        Create an instance of the Analytics Framework Service with the
        typical starting settings.

        If not provided, will reuse the runtime client for the ui
        as well.
        """
        assert isinstance(asset, predix.admin.asset.Asset), \
            "Require an existing predix.admin.asset.Asset instance"
        assert isinstance(timeseries, predix.admin.timeseries.TimeSeries), \
            "Require an existing predix.admin.timeseries.TimeSeries instance"

        parameters = {
            'predixAssetZoneId': asset.get_zone_id(),
            'predixTimeseriesZoneId': timeseries.get_query_zone_id(),
            'runtimeClientId': client_id,
            'runtimeClientSecret': client_secret,
            'uiClientId': ui_client_id or client_id,
            'uiClientSecret': ui_client_secret or client_secret,
            'uiDomainPrefix': self.service.name,
        }

        self.service.create(parameters=parameters)