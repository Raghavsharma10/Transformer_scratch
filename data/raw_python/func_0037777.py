def dump_commands(self, commands):
        """Dump commands wrapper for external access."""
        # Get base directory
        directory = os.path.join(os.path.dirname(self.sql_script), 'fails')

        # Get file name to be used for folder name
        fname = os.path.basename(self.sql_script.rsplit('.')[0])

        return dump_commands(commands, directory, fname)