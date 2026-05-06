def send(self, commands):
        "Ship commands to the daemon."
        if not commands.endswith("\n"):
            commands += "\n"
        self.sock.send(commands)