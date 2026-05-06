def pass_control_back(self, primary, secondary):
        """The address to which the controll is to be passed back.

        Tells a potential controller device the address to which the control is
        to be passed back.

        :param primary: An integer in the range 0 to 30 representing the
            primary address of the controller sending the command.
        :param secondary: An integer in the range of 0 to 30 representing the
            secondary address of the controller sending the command. If it is
            missing, it indicates that the controller sending this command does
            not have extended addressing.

        """
        if secondary is None:
            self._write(('*PCB', Integer(min=0, max=30)), primary)
        else:
            self._write(
                ('*PCB', [Integer(min=0, max=30), Integer(min=0, max=30)]),
                primary,
                secondary
            )