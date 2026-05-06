def _set_relationship_type(self, type_identifier, display_name=None, display_label=None, description=None, domain='Relationship'):
        """Sets the relationship type"""
        if display_name is None:
            display_name = type_identifier
        if display_label is None:
            display_label = display_name
        if description is None:
            description = 'Relationship Type for ' + display_name
        self._relationship_type = Type(authority='DLKIT',
                                       namespace='relationship.Relationship',
                                       identifier=type_identifier,
                                       display_name=display_name,
                                       display_label=display_label,
                                       description=description,
                                       domain=domain)