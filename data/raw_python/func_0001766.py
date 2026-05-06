def _tolog(self,level):
        """ log with different level """
        def wrapper(msg):
            if self.log_colors:
                color = self.log_colors[level.upper()]
                getattr(self.logger, level.lower())(coloring("- {}".format(msg), color))
            else:
                getattr(self.logger, level.lower())(msg)
    
        return wrapper