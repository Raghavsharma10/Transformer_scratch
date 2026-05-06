def get_all_resource_ids_in_datastore(configuration=None):
        # type: (Optional[Configuration]) -> List[str]
        """Get list of resources that have a datastore returning their ids.

        Args:
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.

        Returns:
            List[str]: List of resource ids that are in the datastore
        """
        resource = Resource(configuration=configuration)
        success, result = resource._read_from_hdx('datastore', '_table_metadata', 'resource_id',
                                                  Resource.actions()['datastore_search'], limit=10000)
        resource_ids = list()
        if not success:
            logger.debug(result)
        else:
            for record in result['records']:
                resource_ids.append(record['name'])
        return resource_ids