def auth_criteria(self):
        """
            This attribute provides the mapping of services to their auth requirement

            Returns:
                (dict) : the mapping from services to their auth requirements.
        """
        # the dictionary we will return
        auth = {}

        # go over each attribute of the service
        for attr in dir(self):
            # make sure we could hit an infinite loop
            if attr != 'auth_criteria':
                # get the actual attribute
                attribute = getattr(self, attr)
                # if the service represents an auth criteria
                if isinstance(attribute, Callable) and hasattr(attribute, '_service_auth'):
                    # add the criteria to the final results
                    auth[getattr(self, attr)._service_auth] = attribute

        # return the auth mapping
        return auth