def set_field(self, target, rate, wait_for_stability=True):
        """Sets the field to the specified value.
        
        :param field: The target field in Tesla.
        :param rate: The field rate in tesla per minute.
        :param wait_for_stability: If True, the function call blocks until the
            target field is reached.
        
        """
        self.field.target = target
        self.field.sweep_rate = rate
        self.activity = 'to setpoint'
        while self.status['mode'] != 'at rest':
            time.sleep(1)