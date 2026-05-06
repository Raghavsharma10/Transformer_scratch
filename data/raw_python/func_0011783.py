def update(self, currentTemp, targetTemp):
        """Calculate PID output value for given reference input and feedback."""
        # in this implementation, ki includes the dt multiplier term,
        # and kd includes the dt divisor term.  This is typical practice in
        # industry.
        self.targetTemp = targetTemp
        self.error = targetTemp - currentTemp

        self.P_value = self.Kp * self.error
        # it is common practice to compute derivative term against PV,
        # instead of de/dt.  This is because de/dt spikes
        # when the set point changes.

        # PV version with no dPV/dt filter - note 'previous'-'current',
        # that's desired, how the math works out
        self.D_value = self.Kd * (self.Derivator - currentTemp)
        self.Derivator = currentTemp

        self.Integrator = self.Integrator + self.error
        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min

        self.I_value = self.Integrator * self.Ki

        output = self.P_value + self.I_value + self.D_value
        if output > self.Output_max:
            output = self.Output_max
        if output < self.Output_min:
            output = self.Output_min
        return(output)