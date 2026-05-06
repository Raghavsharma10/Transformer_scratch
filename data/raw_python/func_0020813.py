def listBlockParents(self, **kwargs):
        """
        API to list block parents.

        :param block_name: name of block who's parents needs to be found (Required)
        :type block_name: str
        :returns: List of dictionaries containing following keys (block_name)
        :rtype: list of dicts
       
        """
        validParameters = ['block_name']

        requiredParameters = {'forced': validParameters}
        checkInputParameter(method="listBlockParents", parameters=kwargs.keys(), validParameters=validParameters,
                            requiredParameters=requiredParameters)
        if isinstance(kwargs["block_name"], list):
            return self.__callServer("blockparents", data=kwargs, callmethod='POST')
        else:
            return self.__callServer("blockparents", params=kwargs)