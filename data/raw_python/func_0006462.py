def _get_plsr_dac_charge(self, plsr_dac_array, no_offset=False):
        '''Takes the PlsrDAC calibration and the stored C-high/C-low mask to calculate the charge from the PlsrDAC array on a pixel basis
        '''
        charge = np.zeros_like(self.c_low_mask, dtype=np.float16)  # charge in electrons
        if self.vcal_c0 is not None and self.vcal_c1 is not None and self.c_low is not None and self.c_mid is not None and self.c_high is not None:
            voltage = self.vcal_c1 * plsr_dac_array if no_offset else self.vcal_c0 + self.vcal_c1 * plsr_dac_array
            charge[np.logical_and(self.c_low_mask, ~self.c_high_mask)] = voltage[np.logical_and(self.c_low_mask, ~self.c_high_mask)] * self.c_low / 0.16022
            charge[np.logical_and(~self.c_low_mask, self.c_high_mask)] = voltage[np.logical_and(self.c_low_mask, ~self.c_high_mask)] * self.c_mid / 0.16022
            charge[np.logical_and(self.c_low_mask, self.c_high_mask)] = voltage[np.logical_and(self.c_low_mask, ~self.c_high_mask)] * self.c_high / 0.16022
        return charge