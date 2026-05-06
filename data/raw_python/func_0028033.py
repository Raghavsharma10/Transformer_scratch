def get_all_organization_names(configuration=None, **kwargs):
        # type: (Optional[Configuration], Any) -> List[str]
        """Get all organization names in HDX

        Args:
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.
            **kwargs: See below
            sort (str): Sort the search results according to field name and sort-order. Allowed fields are ‘name’, ‘package_count’ and ‘title’. Defaults to 'name asc'.
            organizations (List[str]): List of names of the groups to return.
            all_fields (bool): Return group dictionaries instead of just names. Only core fields are returned - get some more using the include_* options. Defaults to False.
            include_extras (bool): If all_fields, include the group extra fields. Defaults to False.
            include_tags (bool): If all_fields, include the group tags. Defaults to False.
            include_groups: If all_fields, include the groups the groups are in. Defaults to False.

        Returns:
            List[str]: List of all organization names in HDX
        """
        organization = Organization(configuration=configuration)
        organization['id'] = 'all organizations'  # only for error message if produced
        return organization._write_to_hdx('list', kwargs, 'id')