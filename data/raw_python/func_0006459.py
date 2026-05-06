def max_tot_value(self, value):
        """Set maximum ToT value that is considered to be a hit"""
        self._max_tot_value = value
        self.interpreter.set_max_tot(self._max_tot_value)
        self.histogram.set_max_tot(self._max_tot_value)
        self.clusterizer.set_max_hit_charge(self._max_tot_value)