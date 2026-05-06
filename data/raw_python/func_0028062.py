def search_in_hdx(query, configuration=None, **kwargs):
        # type: (str, Optional[Configuration], Any) -> List['Resource']
        """Searches for resources in HDX. NOTE: Does not search dataset metadata!

        Args:
            query (str): Query
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.
            **kwargs: See below
            order_by (str): A field on the Resource model that orders the results
            offset (int): Apply an offset to the query
            limit (int): Apply a limit to the query
        Returns:
            List[Resource]: List of resources resulting from query
        """

        resources = []
        resource = Resource(configuration=configuration)
        success, result = resource._read_from_hdx('resource', query, 'query', Resource.actions()['search'])
        if result:
            count = result.get('count', None)
            if count:
                for resourcedict in result['results']:
                    resource = Resource(resourcedict, configuration=configuration)
                    resources.append(resource)
        else:
            logger.debug(result)
        return resources