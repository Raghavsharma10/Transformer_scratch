def setPoint(self, targetTemp):
        """Initilize the setpoint of PID."""
        self.targetTemp = targetTemp
        self.Integrator = 0
        self.Derivator = 0