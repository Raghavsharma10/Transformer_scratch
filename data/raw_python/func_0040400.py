def install(self):
        """
        install the server
        """
        try:
            if self.args.server is not None:
                server = ServerLists(self.server_type)
                DynamicImporter(
                    'ezhost',
                    server.name,
                    args=self.args,
                    configure=self.configure
                )
            else:
                ServerCommand(self.args)
        except Exception as e:
            raise e