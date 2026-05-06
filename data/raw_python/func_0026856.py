def power_down(self):
        """
        turn off the HX711
        :return: always True
        :rtype bool
        """
        GPIO.output(self._pd_sck, False)
        GPIO.output(self._pd_sck, True)
        time.sleep(0.01)
        return True