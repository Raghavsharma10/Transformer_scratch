def request(self, command_string):
        """ Request """

        self.send(command_string)
        if self.debug:
            print("Telnet Request:  %s" % (command_string))
        while True:
            response = urllib.parse.unquote(self.tn.read_until(b"\n").decode())
            if "success" in response:   # Normal successful reply
                break
            if "huh" in response:       # Something went wrong
                break
            if "connect" in response:   # Special reply to "hello"
                break
            # TODO Keep track of which screen is displayed
            # Try again if response was key, menu or visibility notification.
        if "huh" in response or self.debug:
            print("Telnet Response: %s" % (response[:-1]))
        return response