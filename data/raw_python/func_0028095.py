def add_update_resources(self, resources, ignore_datasetid=False):
        # type: (List[Union[hdx.data.resource.Resource,Dict,str]], bool) -> None
        """Add new or update existing resources with new metadata to the dataset

        Args:
            resources (List[Union[hdx.data.resource.Resource,Dict,str]]): A list of either resource ids or resources metadata from either Resource objects or dictionaries
            ignore_datasetid (bool): Whether to ignore dataset id in the resource. Defaults to False.

        Returns:
            None
        """
        if not isinstance(resources, list):
            raise HDXError('Resources should be a list!')
        for resource in resources:
            self.add_update_resource(resource, ignore_datasetid)