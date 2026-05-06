def set_pwm_frequency(self, value):
        """Set the frequency for all PWM output

        :param value: the frequency in Hz
        """
        self.__check_range('pwm_frequency', value)
        reg_val = self.calc_pre_scale(value)
        logger.debug("Calculated prescale value is %s" % reg_val)
        self.sleep()
        self.write(Registers.PRE_SCALE, reg_val)
        self.wake()