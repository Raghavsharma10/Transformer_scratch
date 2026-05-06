def sense(self):
        """ Launch a command in the 'senses' List, and update the current state."""

        cmd_name = random.choice(self.senses)
        param = ''
        if cmd_name == 'ls':
            if random.randint(0, 1):
                param = '-l'
        elif cmd_name == 'uname':
            # Choose options from predefined ones
            opts = 'asnrvmpio'
            start = random.randint(0, len(opts) - 2)
            end = random.randint(start + 1, len(opts) - 1)
            param = '-{}'.format(opts[start:end])
        command = getattr(self, cmd_name)
        command(param)