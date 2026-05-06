def handleException(self, exc_type, exc_param, exc_tb):
        "Exception handler (False to abort)"
        self.writeline(''.join( traceback.format_exception(exc_type, exc_param, exc_tb) ))
        return True