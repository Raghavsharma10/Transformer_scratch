def executeCommand(self, command):
        """Send Action to Asterisk Manager Interface to execute CLI Command.
        
        @param command: CLI command to execute.
        @return:        Command response string.

        """
        self._sendAction("Command", (
            ("Command", command),
        ))
        resp = self._getResponse()
        result = resp.get("Response")
        if result == "Follows":
            return resp.get("command_response")
        elif result == "Error":
            raise Exception("Execution of Asterisk Manager Interface Command "
                            "(%s) failed with error message: %s" % 
                            (command, str(resp.get("Message"))))
        else:
            raise Exception("Execution of Asterisk Manager Interface Command "
                            "failed: %s" % command)