def cmdHISTORY(self, params):
        """
        Display the command history
        """
        cnt = 0
        self.writeline('Command history\n')
        for line in self.history:
            cnt = cnt + 1
            self.writeline("%-5d : %s" % (cnt, ''.join(line)))