def enqueue_command(self, command_name, args, options):
        """Enqueue a new command into this pipeline."""
        assert_open(self)
        promise = Promise()
        self.commands.append((command_name, args, options, promise))
        return promise