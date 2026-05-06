def decide(self):
        """
            Decides the next command to be launched based on the current state.

        :return: Tuple containing the next command name, and it's parameters.
        """
        next_command_name = random.choice(self.COMMAND_MAP[self.state['last_command']])
        param = ''
        if next_command_name == 'retrieve':
            param = random.choice(self.state['file_list'])
        elif next_command_name == 'cwd':
            param = random.choice(self.state['dir_list'])
        return next_command_name, param