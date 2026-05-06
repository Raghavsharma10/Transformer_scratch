def updateAcqEraEndDate(self, **kwargs):
        """
        API to update the end_date of an acquisition era

        :param acquisition_era_name: acquisition_era_name to update (Required)
        :type acquisition_era_name: str
        :param end_date: end_date not zero (Required)
        :type end_date: int

        """
        validParameters = ['end_date', 'acquisition_era_name']

        requiredParameters = {'forced': validParameters}

        checkInputParameter(method="updateAcqEraEndDate", parameters=kwargs.keys(), validParameters=validParameters,
                            requiredParameters=requiredParameters)

        return self.__callServer("acquisitioneras", params=kwargs, callmethod='PUT')