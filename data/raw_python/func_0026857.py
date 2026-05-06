def power_up(self):
        """
        power up the HX711

        :return: always True
        :rtype bool
        """
        GPIO.output(self._pd_sck, False)
        time.sleep(0.01)
        return True