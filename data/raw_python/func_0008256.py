def checkUpdate(self, *args):
        """
        Updates values after first checking instrument parameters are OK.
        This is not integrated within update to prevent ifinite recursion
        since update gets called from ipars.
        """
        g = get_root(self).globals
        if not self.check():
            g.clog.warn('Current observing parameters are not valid.')
            return False

        if not g.ipars.check():
            g.clog.warn('Current instrument parameters are not valid.')
            return False