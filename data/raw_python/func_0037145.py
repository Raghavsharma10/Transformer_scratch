def update_soil_moisture(self):
        """ Method will select the ADC module,
                turn on the analog sensor, wait for voltage settle, 
                and then digitize the sensor voltage. 
            Voltage division/signal loss is accounted for by 
                scaling up the sensor output.
                This may need to be adjusted if a different sensor is used
        """
        SensorCluster.analog_sensor_power(SensorCluster.bus, "on")  # turn on sensor
        sleep(.2)
        TCA_select(SensorCluster.bus, self.mux_addr, SensorCluster.adc_chan)
        moisture = get_ADC_value(
            SensorCluster.bus, SensorCluster.adc_addr, SensorCluster.moisture_chan)
        status = TCA_select(SensorCluster.bus, self.mux_addr, "off")  # Turn off mux.
        SensorCluster.analog_sensor_power(SensorCluster.bus, "off")  # turn off sensor
        if (moisture >= 0):
            soil_moisture = moisture/2.048 # Scale to a percentage value 
            self.soil_moisture = round(soil_moisture,3)
        else:
            raise SensorError(
                "The soil moisture meter is not configured correctly.")
        return status