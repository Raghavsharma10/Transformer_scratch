def get_energy(self):
        """
        Returns the consumed energy since the start of the statistics in Wh.
        Attention: Returns None if the value can't be queried or is unknown.
        """
        value = self.box.homeautoswitch("getswitchenergy", self.actor_id)
        return int(value) if value.isdigit() else None