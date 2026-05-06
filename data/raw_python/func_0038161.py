def error_handler(task):
    """Handle and log RPC errors."""
    @wraps(task)
    def wrapper(self, *args, **kwargs):
        try:
            return task(self, *args, **kwargs)
        except Exception as e:
            self.connected = False
            if not self.testing:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                error_message = (
                    "[" + str(datetime.now()) + "] Error in task \"" +
                    task.__name__ + "\" (" +
                    fname + "/" + str(exc_tb.tb_lineno) +
                    ")" + e.message
                )
                self.logger.error("%s: RPC instruction failed" % error_message)
    return wrapper