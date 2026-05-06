def get_description(self, path, method, view):
        """
        Determine a link description.

        This will be based on the method docstring if one exists,
        or else the class docstring.
        """
        description = super(WaldurSchemaGenerator, self).get_description(path, method, view)

        permissions_description = get_permissions_description(view, method)
        if permissions_description:
            description += '\n\n' + permissions_description if description else permissions_description

        if isinstance(view, core_views.ActionsViewSet):
            validators_description = get_validators_description(view)
            if validators_description:
                description += '\n\n' + validators_description if description else validators_description

        validation_description = get_validation_description(view, method)
        if validation_description:
            description += '\n\n' + validation_description if description else validation_description

        return description