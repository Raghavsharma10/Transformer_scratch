def call(self, command, *args):
        """
        Passes an arbitrary command to the coin daemon.

        Args:
          command (str): command to be sent to the coin daemon

        """
        return self.rpc.call(str(command), *args)