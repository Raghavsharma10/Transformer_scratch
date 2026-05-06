def handle(self):
        "The actual service to which the user has connected."
        if not self.authentication_ok():
            return
        if self.DOECHO:
            self.writeline(self.WELCOME)
        self.session_start()
        while self.RUNSHELL:
            read_line = self.readline(prompt=self.PROMPT).strip('\r\n')
            if read_line:
                self.session.transcript_incoming(read_line)
            self.input = self.input_reader(self, read_line)
            self.raw_input = self.input.raw
            if self.input.cmd:
                # TODO: Command should not be converted to upper
                # looks funny in error messages.
                cmd = self.input.cmd.upper()
                params = self.input.params
                if cmd in self.COMMANDS:
                    try:
                        self.COMMANDS[cmd](params)
                    except:
                        logger.exception('Error calling {0}.'.format(cmd))
                        (t, p, tb) = sys.exc_info()
                        if self.handleException(t, p, tb):
                            break
                else:
                    self.writeline('-bash: {0}: command not found'.format(cmd))
                    logger.error("Unknown command '{0}'".format(cmd))
        logger.debug("Exiting handler")