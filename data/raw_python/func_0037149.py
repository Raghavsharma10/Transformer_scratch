def get_water_level(cls):
        """ This method uses the ADC on the control module to measure
            the current water tank level and returns the water volume
            remaining in the tank.

            For this method, it is assumed that a simple voltage divider
            is used to interface the sensor to the ADC module.
            
            Testing shows that the sensor response is not completely linear,
                though it is quite close. To make the results more accurate,
                a mapping method approximated by a linear fit to data is used.
        """
        # ----------
        # These values should be updated based on the real system parameters
        vref = 4.95
        tank_height = 17.5 # in centimeters (height of container)
        rref = 2668  # Reference resistor
        # ----------
        val = 0
        for i in range(5):
            # Take five readings and do an average
            # Fetch value from ADC (0x69 - ch1)
            val = get_ADC_value(cls.bus, 0x6c, 1) + val
        avg = val / 5
        water_sensor_res = rref * avg/(vref - avg)
        depth_cm = water_sensor_res * \
                    (-.0163) + 28.127 # measured transfer adjusted offset
        if depth_cm < 1.0: # Below 1cm, the values should not be trusted.
            depth_cm = 0
        cls.water_remaining = depth_cm / tank_height
        # Return the current depth in case the user is interested in
        #   that parameter alone. (IE for automatic shut-off)
        return depth_cm/tank_height