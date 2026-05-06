def sense(self):
        """
            Launches a few "sensing" commands such as 'ls', or 'pwd'
            and updates the current bait state.
        """
        cmd_name = random.choice(self.senses)
        command = getattr(self, cmd_name)
        self.state['last_command'] = cmd_name
        command()