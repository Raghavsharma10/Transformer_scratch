def get_type_data(self, name):
        """Return dictionary representation of type."""
        return {
            'authority': 'birdland.mit.edu',
            'namespace': 'Genus Types',
            'identifier': name,
            'domain': 'Generic Types',
            'display_name': self.generic_types[name] + ' Generic Type',
            'display_label': self.generic_types[name],
            'description': ('The ' + self.generic_types[name] +
                            ' Type. This type has no symantic meaning.')
        }