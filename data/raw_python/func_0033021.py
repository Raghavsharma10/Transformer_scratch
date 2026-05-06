def scan_temperature(self, measure, temperature, rate, delay=1):
        """Performs a temperature scan.

        Measures until the target temperature is reached.

        :param measure: A callable called repeatedly until stability at target
            temperature is reached.
        :param temperature: The target temperature in kelvin.
        :param rate: The sweep rate in kelvin per minute.
        :param delay: The time delay between each call to measure in seconds.

        """
        # set target temperature to current control temperature
        self.target_temperature = Tset =  self.control_temperature
        # we use a positive sign for the sweep rate if we sweep up and negative
        # if we sweep down.
        rate = abs(rate) if temperature - Tset > 0 else -abs(rate)
        
        t_last = time.time()
        time.sleep(delay)
        while True:
            measure()
            
            # Update setpoint
            t_now = time.time()
            dt = t_now - t_last
            dT =  dt * rate / 60.
            t_last = t_now
            if abs(temperature - Tset) < abs(dT):
                self.target_temperature = temperature
                break
            else:
                self.target_temperature = Tset = Tset + dT
            time.sleep(delay)