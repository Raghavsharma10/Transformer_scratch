def parse(self, commands):
        """ Parse a list of commands.
        """

        # Get rid of dummy objects that represented deleted objects in
        # the last parsing round.
        to_delete = []
        for id_, val in self._objects.items():
            if val == JUST_DELETED:
                to_delete.append(id_)
        for id_ in to_delete:
            self._objects.pop(id_)

        for command in commands:
            self._parse(command)