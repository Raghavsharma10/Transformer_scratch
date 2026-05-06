def roll_dice(self):  # Roll dice with current roll
        """
        Rolls dicebag and sets last_roll and last_explanation to roll results

        :return: Roll results.
        """
        roll = roll_dice(self.roll, floats=self.floats, functions=self.functions)

        self._last_roll = roll[0]
        self._last_explanation = roll[1]

        return self.last_roll, self.last_explanation