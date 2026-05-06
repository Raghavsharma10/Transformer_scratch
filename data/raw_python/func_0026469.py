def ready(self):
        """Sets up the application after startup."""

        self.log('Got', len(schemastore), 'data and',
                 len(configschemastore), 'component schemata.', lvl=debug)