def _update_B(self):
        """Update `B`."""
        for param in self.freeparams:
            if param == 'mu':
                continue
            paramval = getattr(self, param)
            assert isinstance(paramval, float), "Paramvalues must be floats"
            self.B[param] = broadcastMatrixMultiply(self.Ainv,
                    broadcastMatrixMultiply(self.dPxy[param], self.A))