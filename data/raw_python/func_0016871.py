def frequency(self, context):
        """ Frequency data source """
        channels = self._manager.spectral_window_table.getcol(MS.CHAN_FREQ)
        return channels.reshape(context.shape).astype(context.dtype)