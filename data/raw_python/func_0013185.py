def set_range(self, value):
        """Set the range of the accelerometer to the provided value.  Range value
        should be one of these constants:
          - ADXL345_RANGE_2_G   = +/-2G
          - ADXL345_RANGE_4_G   = +/-4G
          - ADXL345_RANGE_8_G   = +/-8G
          - ADXL345_RANGE_16_G  = +/-16G
        """
        # Read the data format register to preserve bits.  Update the data
        # rate, make sure that the FULL-RES bit is enabled for range scaling
        format_reg = self._device.readU8(ADXL345_REG_DATA_FORMAT) & ~0x0F
        format_reg |= value
        format_reg |= 0x08  # FULL-RES bit enabled
        # Write the updated format register.
        self._device.write8(ADXL345_REG_DATA_FORMAT, format_reg)