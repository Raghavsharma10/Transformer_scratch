def doRollover(self):
        """Do a rollover, as described in __init__()."""
        self.stream.close()
        try:
            if self.backupCount > 0:
                tmp_location = "%s.0" % self.baseFilename
                os.rename(self.baseFilename, tmp_location)
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = "%s.%d" % (self.baseFilename, i)
                    dfn = "%s.%d" % (self.baseFilename, i + 1)
                    if os.path.exists(sfn):
                        if os.path.exists(dfn):
                            os.remove(dfn)
                        os.rename(sfn, dfn)
                dfn = self.baseFilename + ".1"
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(tmp_location, dfn)
        except Exception:
            pass
        finally:
            self.stream = WindowsFile(self.baseFilename, "a", self.encoding)