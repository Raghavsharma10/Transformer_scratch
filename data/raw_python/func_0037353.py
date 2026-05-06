def update(self):
        """ This method exposes a more simple interface to the IO module
        Regardless of what the control instance contains, this method
        will transmit the queued IO commands to the IO expander

        Usage: plant1Control.update(bus)
        """
        ControlCluster.compile_instance_masks()

        IO_expander_output(
            ControlCluster.bus, self.IOexpander,
            self.bank,
            ControlCluster.master_mask[self.bank])

        if self.bank != ControlCluster.pump_bank:
            IO_expander_output(
                ControlCluster.bus, self.IOexpander,
                ControlCluster.pump_bank,
                ControlCluster.master_mask[ControlCluster.pump_bank])