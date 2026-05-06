def _filter(self, commands, parser):
        """ Filter DATA/SIZE commands that are overridden by a
        SIZE command.
        """
        resized = set()
        commands2 = []
        for command in reversed(commands):
            if command[0] == 'SHADERS':
                convert = parser.convert_shaders()
                if convert:
                    shaders = self._convert_shaders(convert, command[2:])
                    command = command[:2] + shaders
            elif command[1] in resized:
                if command[0] in ('SIZE', 'DATA'):
                    continue  # remove this command
            elif command[0] == 'SIZE':
                resized.add(command[1])
            commands2.append(command)
        return list(reversed(commands2))