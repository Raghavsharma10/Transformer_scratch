def get_pwm(self, led_num):
        """Generic getter for all LED PWM value"""
        self.__check_range('led_number', led_num)
        register_low = self.calc_led_register(led_num)
        return self.__get_led_value(register_low)