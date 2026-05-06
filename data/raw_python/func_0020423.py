def get_power(self):
        """
        Returns the current power usage in milliWatts.
        Attention: Returns None if the value can't be queried or is unknown.
        """
        value = self.box.homeautoswitch("getswitchpower", self.actor_id)
        return int(value) if value.isdigit() else None