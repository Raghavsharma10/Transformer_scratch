def analog_sensor_power(cls, bus, operation):
        """ Method that turns on all of the analog sensor modules
            Includes all attached soil moisture sensors
            Note that all of the SensorCluster object should be attached
                in parallel and only 1 GPIO pin is available
                to toggle analog sensor power.
            The sensor power should be left on for at least 100ms
                in order to allow the sensors to stabilize before reading. 
                Usage:  SensorCluster.analog_sensor_power(bus,"high")
                OR      SensorCluster.analog_sensor_power(bus,"low")
            This method should be removed if an off-board GPIO extender is used.
        """
        # Set appropriate analog sensor power bit in GPIO mask
        # using the ControlCluster bank_mask to avoid overwriting any data
        reg_data = get_IO_reg(bus, 0x20, cls.power_bank)

        if operation == "on":
            reg_data = reg_data | 1 << cls.analog_power_pin
        elif operation == "off":
            reg_data = reg_data & (0b11111111 ^ (1 << cls.analog_power_pin))
        else:
            raise SensorError(
                "Invalid command used while enabling analog sensors")
        # Send updated IO mask to output
        IO_expander_output(bus, 0x20, cls.power_bank, reg_data)