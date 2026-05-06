def reload_using_spawn_wait(self):
        """
        Spawn a subprocess and wait until it finishes.

        :return:
            None.
        """
        # Create command parts
        cmd_parts = [sys.executable] + sys.argv

        # Get env dict copy
        env_copy = os.environ.copy()

        # Send interrupt to main thread
        interrupt_main()

        # Spawn subprocess and wait until it finishes
        subprocess.call(cmd_parts, env=env_copy, close_fds=True)

        # Exit the watcher thread
        sys.exit(0)