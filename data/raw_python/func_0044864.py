def sampling_volume_value(self):
        """Returns the device samping volume value in m."""
        svi = self.pdx.SamplingVolume
        tli = self.pdx.TransmitLength
        return self._sampling_volume_value(svi, tli)