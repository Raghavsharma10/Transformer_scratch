def _prepare_wsdl_objects(self):
        """
        Create the data structure and get it ready for the WSDL request.
        """

        # Service defaults for objects that are required.
        self.MultipleMatchesAction = 'RETURN_ALL'
        self.Constraints = self.create_wsdl_object_of_type('SearchLocationConstraints')
        self.Address = self.create_wsdl_object_of_type('Address')
        self.LocationsSearchCriterion = 'ADDRESS'
        self.SortDetail = self.create_wsdl_object_of_type('LocationSortDetail')