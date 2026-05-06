def get_add_files_command(self, *filenames):
        """Get the command to include added and/or removed files in the working tree in the next commit."""
        command = ['git', 'add']
        if filenames:
            command.append('--')
            command.extend(filenames)
        else:
            command.extend(('--all', '.'))
        return command