def log_error(self, error, message, detail=None, strip=4):
        "Add an error message and optional user message to the error list"
        if message:
            msg = message + ": " + error
        else:
            msg = error

        tb = traceback.format_stack()
        if sys.version_info >= (3, 0):
            tb = tb[:-strip]
        else:
            tb = tb[strip:]

        self.errors.append({
            'message': msg,
            'traceback': tb,
            'detail': detail
        })