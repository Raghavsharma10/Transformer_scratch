def actions(self):
        """A list of actions to perform.

        :return: a list of :class:`AYABInterface.actions.Action`
        """
        actions = []
        do = actions.append

        # determine the number of colors
        colors = self.colors

        # rows and colors
        movements = (
            MoveCarriageToTheRight(KnitCarriage()),
            MoveCarriageToTheLeft(KnitCarriage()))
        rows = self._rows
        first_needles = self._get_row_needles(0)

        # handle switches
        if len(colors) == 1:
            actions.extend([
                SwitchOffMachine(),
                SwitchCarriageToModeNl()])
        else:
            actions.extend([
                SwitchCarriageToModeKc(),
                SwitchOnMachine()])

        # move needles
        do(MoveNeedlesIntoPosition("B", first_needles))
        do(MoveCarriageOverLeftHallSensor())

        # use colors
        if len(colors) == 1:
            do(PutColorInNutA(colors[0]))
        if len(colors) == 2:
            do(PutColorInNutA(colors[0]))
            do(PutColorInNutB(colors[1]))

        # knit
        for index, row in enumerate(rows):
            do(movements[index & 1])
        return actions