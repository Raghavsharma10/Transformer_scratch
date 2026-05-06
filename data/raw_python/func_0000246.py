def getPayloadStruct(self, attributes, objType):
        """ Function getPayloadStruct
        Get the payload structure to do a creation or a modification

        @param attribute: The data
        @param objType: SubItem type (e.g: hostgroup for hostgroup_class)
        @return RETURN: the payload
        """
        payload = {self.payloadObj: attributes,
                   objType + "_class":
                       {self.payloadObj: attributes}}
        return payload