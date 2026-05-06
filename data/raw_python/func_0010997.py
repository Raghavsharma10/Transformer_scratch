def roll(self, value):
        """
        Setter for roll, verifies the roll is valid

        :param value: Roll
        :return: None
        """
        if type(value) != str:  # Make sure dice roll is a str
            raise TypeError('Dice roll must be a string in dice notation')
        try:
            roll_dice(value)  # Make sure dice roll parses as a valid roll and not an error
        except Exception as e:
            raise ValueError('Dice roll specified was not a valid diceroll.\n%s\n' % str(e))
        else:
            self._roll = value