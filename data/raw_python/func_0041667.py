def signal_handler(self, signum, frame):
        """
        Handle print_exit via signals.
        """
        self.print_exit()
        print("\n(Terminated with signal %d)\n" % (signum))
        sys.exit(0)