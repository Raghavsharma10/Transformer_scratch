def user_method(user_event):
    """Decorator of the Pdb user_* methods that controls the RemoteSocket."""
    def wrapper(self, *args):
        stdin = self.stdin
        is_sock = isinstance(stdin, RemoteSocket)
        try:
            try:
                if is_sock and not stdin.connect():
                    return
                return user_event(self, *args)
            except Exception:
                self.close()
                raise
        finally:
            if is_sock and stdin.closed():
                self.do_detach(None)
    return wrapper