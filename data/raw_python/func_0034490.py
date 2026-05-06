def raise_and_log_error(self, error, message):
        """Raise error, including message and original traceback.

        error: the error to raise
        message: the user-facing error message
        """
        self.log('raising %s, traceback %s\n' %
                 (error, traceback.format_exc()))
        raise error(message)