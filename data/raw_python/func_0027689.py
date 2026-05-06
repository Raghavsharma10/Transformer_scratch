def runcode(self, code):
        """
        Override L{code.InteractiveConsole.runcode} to run the code in a
        transaction unless the local C{autocommit} is currently set to a true
        value.
        """
        if not self.locals.get('autocommit', None):
            return self.locals['db'].transact(code.InteractiveConsole.runcode, self, code)
        return code.InteractiveConsole.runcode(self, code)