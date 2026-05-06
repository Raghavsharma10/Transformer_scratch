def listFileParentsByLumi(self, **kwargs):
        """
        API to list file parents using lumi section info.

        :param block_name: name of block that has files who's parents needs to be found (Required)
        :type block_name: str
        :param logical_file_name: if not all the file parentages under the block needed, this lfn list gives the files that needs to find its parents(optional).
        :type logical_file_name: list of string  
        :returns: List of dictionaries containing following keys [cid,pid]
        :rtype: list of dicts
       
        """
        validParameters = ['block_name', 'logical_file_name']

        requiredParameters = {'forced': ['block_name']}
        checkInputParameter(method="listFileParentsByLumi", parameters=kwargs.keys(), validParameters=validParameters,
                            requiredParameters=requiredParameters)
        return self.__callServer("fileparentsbylumi", data=kwargs, callmethod='POST')