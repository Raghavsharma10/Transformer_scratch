def db_access_point(func):
    """
    Wraps a function that actually accesses the database.
    It injects a session into the method and attempts to handle
    it after the function has run.

    :param method func: The method that is interacting with the database.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        Wrapper responsible for handling
        sessions
        """
        session = self.session_handler.get_session()
        try:
            resp = func(self, session, *args, **kwargs)
        except Exception as exc:
            self.session_handler.handle_session(session, exc=exc)
            raise exc
        else:
            self.session_handler.handle_session(session)
            return resp
    return wrapper