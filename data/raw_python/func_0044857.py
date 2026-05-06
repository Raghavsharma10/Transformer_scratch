def transmit_length(self, val=3):
        """Sets transmit length."""
        if self.instrument == "Vectrino" and type(val) is float:
            if val == 0.3:
                self.pdx.TransmitLength = 0
            elif val == 0.6:
                self.pdx.TransmitLength = 1
            elif val == 1.2:
                self.pdx.TransmitLength = 2
            elif val == 1.8:
                self.pdx.TransmitLength = 3
            elif val == 2.4:
                self.pdx.TransmitLength = 4
            else:
                raise ValueError("Invalid transmit length specified")
        elif val in range(5):
            self.pdx.TransmitLength = val
        else:
            raise ValueError("Invalid transmit length specified")