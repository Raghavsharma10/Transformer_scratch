def set_temperature(self, temperature, rate, mode='fast', wait_for_stability=True, delay=1):
        """Sets the temperature.

        :param temperature: The target temperature in kelvin.
        :param rate: The sweep rate in kelvin per minute.
        :param mode: The sweep mode, either 'fast' or 'no overshoot'.
        :param wait_for_stability: If wait_for_stability is `True`, the function call blocks
            until the target temperature is reached and stable.
        :param delay: The delay specifies the frequency how often the status is checked.

        """
        self.target_temperature = temperature, rate, mode
        start = datetime.datetime.now()
        while wait_for_stability:
            # The PPMS needs some time to update the status code, we therefore ignore it for 10s.
            if (self.system_status['temperature'] == 'normal stability at target temperature' and
                (datetime.datetime.now() - start > datetime.timedelta(seconds=10))):
                break
            time.sleep(delay)