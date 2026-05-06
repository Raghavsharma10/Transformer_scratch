def cli_schemata_list(self, *args):
        """Display a list of registered schemata"""

        self.log('Registered schemata languages:', ",".join(sorted(l10n_schemastore.keys())))
        self.log('Registered Schemata:', ",".join(sorted(schemastore.keys())))
        if '-c' in args or '-config' in args:
            self.log('Registered Configuration Schemata:', ",".join(sorted(configschemastore.keys())), pretty=True)