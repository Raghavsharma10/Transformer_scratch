def reload_using_spawn_exit(self):
        """
        Spawn a subprocess and exit the current process.

        :return:
            None.
        """
        # Create command parts
        cmd_parts = [sys.executable] + sys.argv

        # Get env dict copy
        env_copy = os.environ.copy()

        # Spawn subprocess
        subprocess.Popen(cmd_parts, env=env_copy, close_fds=True)

        # If need force exit
        if self._force_exit:
            # Force exit
            os._exit(0)  # pylint: disable=protected-access

        # If not need force exit
        else:
            # Send interrupt to main thread
            interrupt_main()

        # Set the flag
        self._watcher_to_stop = True

        # Exit the watcher thread
        sys.exit(0)