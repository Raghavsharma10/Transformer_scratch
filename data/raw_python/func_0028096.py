def delete_resource(self, resource, delete=True):
        # type: (Union[hdx.data.resource.Resource,Dict,str], bool) -> bool
        """Delete a resource from the dataset and also from HDX by default

        Args:
            resource (Union[hdx.data.resource.Resource,Dict,str]): Either resource id or resource metadata from a Resource object or a dictionary
            delete (bool): Whetehr to delete the resource from HDX (not just the dataset). Defaults to True.

        Returns:
            bool: True if resource removed or False if not
        """
        if isinstance(resource, str):
            if is_valid_uuid(resource) is False:
                raise HDXError('%s is not a valid resource id!' % resource)
        return self._remove_hdxobject(self.resources, resource, delete=delete)