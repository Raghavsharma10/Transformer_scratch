def _run(self):
        """The inside of ``run``'s infinite loop.

        Separated out so it can be properly unit tested.
        """
        cmd = self.read_command()
        if cmd["command"] == "next":
            self.next_tuple()
        elif cmd["command"] == "ack":
            self.ack(cmd["id"])
        elif cmd["command"] == "fail":
            self.fail(cmd["id"])
        elif cmd["command"] == "activate":
            self.activate()
        elif cmd["command"] == "deactivate":
            self.deactivate()
        else:
            self.logger.error("Received invalid command from Storm: %r", cmd)
        self.send_message({"command": "sync"})