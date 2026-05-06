def do_login(self, line):
        "login aws-acces-key aws-secret"
        if line:
            args = self.getargs(line)
            self.connect(args[0], args[1])
        else:
            self.connect()

        self.do_tables('')