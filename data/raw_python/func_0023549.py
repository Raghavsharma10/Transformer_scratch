def do_quit(self, arg):
        """q(uit)\nexit
        Quit from the debugger. The program being executed is aborted.
        """
        if isinstance(self.stdin, RemoteSocket) and not self.is_debug_instance:
            return self.do_detach(arg)
        self._user_requested_quit = True
        self.set_quit()
        return 1