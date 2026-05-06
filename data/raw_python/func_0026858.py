def reset(self):
        """
        reset the HX711 and prepare it for 	the next reading

        :return: True on success
        :rtype bool
        :raises GenericHX711Exception
        """
        logging.debug("power down")
        self.power_down()
        logging.debug("power up")
        self.power_up()
        logging.debug("read some raw data")
        result = self.get_raw_data(6)
        if result is False:
            raise GenericHX711Exception("failed to reset HX711")
        else:
            return True