def insert(self, resourcetype, source, insert_date=None):
        """
        Insert a record into the repository
        """

        caller = inspect.stack()[1][3]

        if caller == 'transaction':  # insert of Layer
            hhclass = 'Layer'
            source = resourcetype
            resourcetype = resourcetype.csw_schema
        else:  # insert of service
            hhclass = 'Service'
            if resourcetype not in HYPERMAP_SERVICE_TYPES.keys():
                raise RuntimeError('Unsupported Service Type')

        return self._insert_or_update(resourcetype, source, mode='insert', hhclass=hhclass)