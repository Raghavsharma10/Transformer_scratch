def get_timeseries(self, *args, **kwargs):
        """
        Returns an instance of the Time Series Service.
        """
        import predix.data.timeseries
        ts = predix.data.timeseries.TimeSeries(*args, **kwargs)
        return ts