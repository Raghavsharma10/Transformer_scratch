def listRuns(self, **kwargs):
        """
        API to list all run dictionary, for example: [{'run_num': [160578, 160498, 160447, 160379]}]. 
        At least one parameter is mandatory.

        :param logical_file_name: List all runs in the file
        :type logical_file_name: str
        :param block_name: List all runs in the block
        :type block_name: str
        :param dataset: List all runs in that dataset
        :type dataset: str
        :param run_num: List all runs
        :type run_num: int, string or list

        """
        validParameters = ['run_num', 'logical_file_name', 'block_name', 'dataset']

        requiredParameters = {'multiple': validParameters}

        checkInputParameter(method="listRuns", parameters=kwargs.keys(), validParameters=validParameters,
                            requiredParameters=requiredParameters)

        return self.__callServer("runs", params=kwargs)