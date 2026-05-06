def set_bounds(self, start, stop):
        """
        Sets boundaries for all instruments in constellation
        """
        for instrument in self.instruments:
            instrument.bounds = (start, stop)