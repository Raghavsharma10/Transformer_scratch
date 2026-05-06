def output(self):
        """Simulated model output"""
        if self._timeseries is None:
            self.compute()
        output = self._timeseries[:, self.system.output_vars]
        if isinstance(self.system, NetworkModel):
            return self.system._reshape_output(output)
        else:
            return output