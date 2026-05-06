def run_commands(self, commands):
        """Only useful for EOS"""
        if "eos" in self.profile:
            return list(self.parent.cli(commands).values())[0]
        else:
            raise AttributeError("MockedDriver instance has not attribute '_rpc'")