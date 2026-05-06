def restore_state(self):
        """ Method should be called on obj. initialization
            When called, the method will attempt to restore 
            IO expander and RPi coherence and restore
            local knowledge across a possible power failure 
        """
        current_mask = get_IO_reg(ControlCluster.bus,
                                 self.IOexpander, 
                                 self.bank)
        if current_mask & (1 << ControlCluster.pump_pin):
            self.manage_pump("on")
        if current_mask & (1 << self.fan):
            self.manage_fan("on")
        if current_mask & (1 << self.light):
            self.manage_fan("on")