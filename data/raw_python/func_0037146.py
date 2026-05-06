def update_instance_sensors(self, opt=None):

        """ Method runs through all sensor modules and updates 
            to the latest sensor values.
        After running through each sensor module,
        The sensor head (the I2C multiplexer), is disabled
        in order to avoid address conflicts.
        Usage:
            plant_sensor_object.updateAllSensors(bus_object)
        """
        self.update_count += 1
        self.update_lux()
        self.update_humidity_temp()
        if opt == "all":
            try:
                self.update_soil_moisture()
            except SensorError:
                # This could be handled with a repeat request later.
                pass
        self.timestamp = time()
        # disable sensor module

        tca_status = TCA_select(SensorCluster.bus, self.mux_addr, "off")
        if tca_status != 0:
            raise I2CBusError(
                "Bus multiplexer was unable to switch off to prevent conflicts")