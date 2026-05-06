def get_temperature(self):
        """
        Returns the current environment temperature.
        Attention: Returns None if the value can't be queried or is unknown.
        """
        #raise NotImplementedError("This should work according to the AVM docs, but don't...")
        value = self.box.homeautoswitch("gettemperature", self.actor_id)
        if value.isdigit():
            self.temperature = float(value)/10
        else:
            self.temperature = None
        return self.temperature