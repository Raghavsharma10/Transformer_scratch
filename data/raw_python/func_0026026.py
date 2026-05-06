def getWithPrompt(self):
        """Interactively prompt for parameter value"""
        if self.prompt:
            pstring = self.prompt.split("\n")[0].strip()
        else:
            pstring = self.name
        if self.choice:
            schoice = list(map(self.toString, self.choice))
            pstring = pstring + " (" + "|".join(schoice) + ")"
        elif self.min not in [None, INDEF] or \
                 self.max not in [None, INDEF]:
            pstring = pstring + " ("
            if self.min not in [None, INDEF]:
                pstring = pstring + self.toString(self.min)
            pstring = pstring + ":"
            if self.max not in [None, INDEF]:
                pstring = pstring + self.toString(self.max)
            pstring = pstring + ")"
        # add current value as default
        if self.value is not None:
            pstring = pstring + " (" + self.toString(self.value,quoted=1) + ")"
        pstring = pstring + ": "
        # don't redirect stdin/out unless redirected filehandles are also ttys
        # or unless originals are NOT ttys
        stdout = sys.__stdout__
        try:
            if sys.stdout.isatty() or not stdout.isatty():
                stdout = sys.stdout
        except AttributeError:
            pass
        stdin = sys.__stdin__
        try:
            if sys.stdin.isatty() or not stdin.isatty():
                stdin = sys.stdin
        except AttributeError:
            pass
        # print prompt, suppressing both newline and following space
        stdout.write(pstring)
        stdout.flush()
        ovalue = irafutils.tkreadline(stdin)
        value = ovalue.strip()
        # loop until we get an acceptable value
        while (1):
            try:
                # null input usually means use current value as default
                # check it anyway since it might not be acceptable
                if value == "": value = self._nullPrompt()
                self.set(value)
                # None (no value) is not acceptable value after prompt
                if self.value is not None: return
                # if not EOF, keep looping
                if ovalue == "":
                    stdout.flush()
                    raise EOFError("EOF on parameter prompt")
                print("Error: specify a value for the parameter")
            except ValueError as e:
                print(str(e))
            stdout.write(pstring)
            stdout.flush()
            ovalue = irafutils.tkreadline(stdin)
            value = ovalue.strip()