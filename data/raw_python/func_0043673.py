def get_export_command(self, directory, revision):
        """Get the command to export the complete tree from the local repository."""
        shell_command = 'git archive %s | tar --extract --directory=%s'
        return [shell_command % (quote(revision), quote(directory))]