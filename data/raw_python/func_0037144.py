def update_humidity_temp(self):
        """ This method utilizes the HIH7xxx sensor to read
            humidity and temperature in one call. 
        """
        # Create mask for STATUS (first two bits of 64 bit wide result)
        STATUS = 0b11 << 6

        TCA_select(SensorCluster.bus, self.mux_addr, SensorCluster.humidity_chan)
        SensorCluster.bus.write_quick(SensorCluster.humidity_addr)  # Begin conversion
        sleep(.25)
        # wait 100ms to make sure the conversion takes place.
        data = SensorCluster.bus.read_i2c_block_data(SensorCluster.humidity_addr, 0, 4)
        status = (data[0] & STATUS) >> 6
        
        if status == 0 or status == 1:  # will always pass for now.
            humidity = round((((data[0] & 0x3f) << 8) |
                              data[1]) * 100.0 / (2**14 - 2), 3)
            self.humidity = humidity
            self.temp = (round((((data[2] << 6) + ((data[3] & 0xfc) >> 2))
                               * 165.0 / 16382.0 - 40.0), 3) * 9/5) + 32
            return TCA_select(SensorCluster.bus, self.mux_addr, "off")
        else:
            raise I2CBusError("Unable to retrieve humidity")