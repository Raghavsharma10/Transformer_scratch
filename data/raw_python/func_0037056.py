def _generate_manager(manager_config):
        '''
        Generate a manager from a manager_config dictionary

        Parameters
        ----------

        manager_config : dict
            Configuration with keys class, args, and kwargs
            used to generate a new datafs.manager object

        Returns
        -------

        manager : object
            datafs.managers.MongoDBManager or
            datafs.managers.DynamoDBManager object
            initialized with *args, **kwargs

        Examples
        --------

        Generate a dynamo manager:

        .. code-block:: python

        >>> mgr = APIConstructor._generate_manager({
        ...     'class': 'DynamoDBManager',
        ...     'kwargs': {
        ...         'table_name': 'data-from-yaml',
        ...         'session_args': {
        ...             'aws_access_key_id': "access-key-id-of-your-choice",
        ...             'aws_secret_access_key': "secret-key-of-your-choice"},
        ...         'resource_args': {
        ...             'endpoint_url':'http://localhost:8000/',
        ...             'region_name':'us-east-1'}
        ...     }
        ... })
        >>>
        >>> from datafs.managers.manager_dynamo import DynamoDBManager
        >>> assert isinstance(mgr, DynamoDBManager)
        >>>
        >>> 'data-from-yaml' in mgr.table_names
        False
        >>> mgr.create_archive_table('data-from-yaml')
        >>> 'data-from-yaml' in mgr.table_names
        True
        >>> mgr.delete_table('data-from-yaml')

        '''

        if 'class' not in manager_config:
            raise ValueError(
                'Manager not fully specified. Give '
                '"class:manager_name", e.g. "class:MongoDBManager".')

        mgr_class_name = manager_config['class']

        if mgr_class_name.lower()[:5] == 'mongo':
            from datafs.managers.manager_mongo import (
                MongoDBManager as mgr_class)

        elif mgr_class_name.lower()[:6] == 'dynamo':
            from datafs.managers.manager_dynamo import (
                DynamoDBManager as mgr_class)

        else:
            raise KeyError(
                'Manager class "{}" not recognized. Choose from {}'.format(
                    mgr_class_name, 'MongoDBManager or DynamoDBManager'))

        manager = mgr_class(
            *manager_config.get('args', []),
            **manager_config.get('kwargs', {}))

        return manager