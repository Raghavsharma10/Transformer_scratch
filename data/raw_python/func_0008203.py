def _CSI(self, cmd):
        """
        Control sequence introducer
        """
        sys.stdout.write('\x1b[')
        sys.stdout.write(cmd)