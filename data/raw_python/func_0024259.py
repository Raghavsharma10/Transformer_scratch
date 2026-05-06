def set_client_cmd(self, *args):
        """
        Adds given cmd(s) to ``self.output['client_cmd']``

        Args:
            *args: Client commands.
        """
        self.client_cmd.update(args)
        self.output['client_cmd'] = list(self.client_cmd)