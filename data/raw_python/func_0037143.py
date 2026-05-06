def update_lux(self, extend=0):
        """ Communicates with the TSL2550D light sensor and returns a 
            lux value. 

        Note that this method contains approximately 1 second of total delay.
            This delay is necessary in order to obtain full resolution
            compensated lux values.

        Alternatively, the device could be put in extended mode, 
            which drops some resolution in favor of shorter delays.

        """
        DEVICE_REG_OUT = 0x1d
        LUX_PWR_ON = 0x03
        if extend == 1:
            LUX_MODE = 0x1d
            delay = .08
            scale = 5
        else:
            LUX_MODE = 0x18
            delay = .4
            scale = 1
        LUX_READ_CH0 = 0x43
        LUX_READ_CH1 = 0x83
        # Select correct I2C mux channel on TCA module

        TCA_select(SensorCluster.bus, self.mux_addr, SensorCluster.lux_chan)
        # Make sure lux sensor is powered up.
        SensorCluster.bus.write_byte(SensorCluster.lux_addr, LUX_PWR_ON)
        lux_on = SensorCluster.bus.read_byte_data(SensorCluster.lux_addr, LUX_PWR_ON)
        
        # Check for successful powerup
        if (lux_on == LUX_PWR_ON):
            # Send command to initiate ADC on each channel
            # Read each channel after the new data is ready
            SensorCluster.bus.write_byte(SensorCluster.lux_addr, LUX_MODE)
            SensorCluster.bus.write_byte(SensorCluster.lux_addr, LUX_READ_CH0)
            sleep(delay)
            adc_ch0 = SensorCluster.bus.read_byte(SensorCluster.lux_addr)
            count0 = get_lux_count(adc_ch0) * scale  # 5x for extended mode
            SensorCluster.bus.write_byte(SensorCluster.lux_addr, LUX_READ_CH1)
            sleep(delay)
            adc_ch1 = SensorCluster.bus.read_byte(SensorCluster.lux_addr)
            count1 = get_lux_count(adc_ch1) * scale  # 5x for extended mode
            ratio = count1 / (count0 - count1)
            lux = (count0 - count1) * .39 * e**(-.181 * (ratio**2))
            self.light_ratio = float(count1)/float(count0)
            print("Light ratio Ch1/Ch0: ", self.light_ratio)
            self.lux = round(lux, 3)
            return TCA_select(SensorCluster.bus, self.mux_addr, "off")
        else:
            raise SensorError("The lux sensor is powered down.")