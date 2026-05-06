def _initialize(self, **resource_attributes):
        """
        Initialize the collection.

        :param resource_attributes: API resource parameters
        """
        super(APIResourceCollection, self)._initialize(**resource_attributes)

        dict_list = self.data
        self.data = []
        for resource in dict_list:
            self.data.append(self._expected_api_resource(**resource))