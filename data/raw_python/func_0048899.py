def __stream_format_allowed(self, stream):
        """
        Check whether a stream allows formatting such as coloring.
        Inspired from Python cookbook, #475186
        """
        # curses isn't available on all platforms
        try:
            import curses as CURSES
        except:
            return False
        try:
            CURSES.setupterm()
            return CURSES.tigetnum("colors") >= 2
        except:
            return False