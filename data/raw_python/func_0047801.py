def get_type_data(self, name):
        """Return dictionary representation of type."""
        try:
            return {
                'authority': 'DLKIT.MIT.EDU',
                'namespace': 'NoneType',
                'identifier': name,
                'domain': 'Generic Types',
                'display_name': self.none_types[name] + ' Type',
                'display_label': self.none_types[name],
                'description': ('The ' + self.none_types[name] +
                                ' Type. This type indicates that no type is specified.')
            }
        except IndexError:
            raise NotFound('NoneType: ' + name)