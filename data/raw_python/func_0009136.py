def _exit(self, status_code):
        """Properly kill Python process including zombie threads."""
        # If there are active threads still running infinite loops, sys.exit
        # won't kill them but os._exit will. os._exit skips calling cleanup
        # handlers, flushing stdio buffers, etc.
        exit_func = os._exit if threading.active_count() > 1 else sys.exit
        exit_func(status_code)