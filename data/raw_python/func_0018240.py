def handle(self):
        "The actual service to which the user has connected."
        if self.TELNET_ISSUE:
            self.writeline(self.TELNET_ISSUE)
        if not self.authentication_ok():
            return
        if self.DOECHO:
            self.writeline(self.WELCOME)

        self.session_start()
        while self.RUNSHELL:
            raw_input = self.readline(prompt=self.PROMPT).strip()
            self.input = self.input_reader(self, raw_input)
            self.raw_input = self.input.raw
            if self.input.cmd:
                cmd = self.input.cmd.upper()
                params = self.input.params
                if self.COMMANDS.has_key(cmd):
                    try:
                        self.COMMANDS[cmd](params)
                    except:
                        log.exception('Error calling %s.' % cmd)
                        (t, p, tb) = sys.exc_info()
                        if self.handleException(t, p, tb):
                            break
                else:
                    self.writeerror("Unknown command '%s'" % cmd)
        log.debug("Exiting handler")