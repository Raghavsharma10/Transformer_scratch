def get_target_temperature(self):
        """
        Returns the actual target temperature.
        Attention: Returns None if the value can't be queried or is unknown.
        """
        value = self.box.homeautoswitch("gethkrtsoll", self.actor_id)
        self.target_temperature = self.__get_temp(value)
        return self.target_temperature