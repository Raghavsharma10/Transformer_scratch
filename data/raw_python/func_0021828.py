def write(self, inline):
        """
        Write a line to stdout if it isn't in a blacklist

        Try to get the name of the calling module to see if we want
        to filter it. If there is no calling module, use current
        frame in case there's a traceback before there is any calling module
        """
        frame = inspect.currentframe().f_back
        if frame:
            mod = frame.f_globals.get('__name__')
        else:
            mod = sys._getframe(0).f_globals.get('__name__')
        if not mod in self.modulenames:
            self.stdout.write(inline)