def seriesshape(self):
        """Shape of the whole time series (time being the first dimension)."""
        seriesshape = [len(hydpy.pub.timegrids.init)]
        seriesshape.extend(self.shape)
        return tuple(seriesshape)