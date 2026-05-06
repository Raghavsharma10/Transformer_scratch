def cli_forms(self, *args):
        """List all available form definitions"""

        forms = []
        missing = []

        for key, item in schemastore.items():
            if 'form' in item and len(item['form']) > 0:
                forms.append(key)
            else:
                missing.append(key)

        self.log('Schemata with form:', forms)
        self.log('Missing forms:', missing)