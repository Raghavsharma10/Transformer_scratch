def compile_instance_masks(cls):
        """ Compiles instance masks into a master mask that is usable by
                the IO expander. Also determines whether or not the pump
                should be on. 
            Method is generalized to support multiple IO expanders
                for possible future expansion.
        """
        # Compute required # of IO expanders needed, clear mask variable.
        number_IO_expanders = ((len(cls._list) - 1) / 4) + 1
        cls.master_mask = [0, 0] * number_IO_expanders

        for ctrlobj in cls:
            # Or masks together bank-by-banl
            cls.master_mask[ctrlobj.bank] |= ctrlobj.mask
            # Handle the pump request seperately
            if ctrlobj.pump_request == 1:
                cls.master_mask[cls.pump_bank] |= 1 << cls.pump_pin