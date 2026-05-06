def reload_using_exec(self):
        """
        Reload the program process.

        :return:
            None.
        """
        # Create command parts
        cmd_parts = [sys.executable] + sys.argv

        # Get env dict copy
        env_copy = os.environ.copy()

        # Reload the program process
        os.execvpe(
            # Program file path
            sys.executable,
            # Command parts
            cmd_parts,
            # Env dict
            env_copy,
        )