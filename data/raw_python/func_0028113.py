def get_all_resources(datasets):
        # type: (List['Dataset']) -> List[hdx.data.resource.Resource]
        """Get all resources from a list of datasets (such as returned by search)

        Args:
            datasets (List[Dataset]): list of datasets

        Returns:
            List[hdx.data.resource.Resource]: list of resources within those datasets
        """
        resources = []
        for dataset in datasets:
            for resource in dataset.get_resources():
                resources.append(resource)
        return resources