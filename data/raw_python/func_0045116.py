def stats(self):
        """ shotcut to pull out useful info for interactive use """
        printDebug("Classes.....: %d" % len(self.classes))
        printDebug("Properties..: %d" % len(self.properties))