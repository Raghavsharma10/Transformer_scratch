def cli_default_perms(self, *args):
        """Show default permissions for all schemata"""

        for key, item in schemastore.items():
            # self.log(item, pretty=True)
            if item['schema'].get('no_perms', False):
                self.log('Schema without permissions:', key)
                continue
            try:
                perms = item['schema']['properties']['perms']['properties']
                if perms == {}:
                    self.log('Schema:', item, pretty=True)

                self.log(
                    'Schema:', key,
                    'read', perms['read']['default'],
                    'write', perms['write']['default'],
                    'list', perms['list']['default'],
                    'create', item['schema']['roles_create']
                )
            except KeyError as e:
                self.log('Fishy schema found:', key, e, lvl=error)
                self.log(item, pretty=True)