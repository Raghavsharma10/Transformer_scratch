def getPayloadStruct(self, attributes, objType=None):
        """ Function getPayloadStruct
        Get the payload structure to do a creation or a modification

        @param key: The key to modify
        @param attribute: The data
        @param objType: NOT USED in this class
        @return RETURN: The API result
        """
        if self.setInParentPayload:
            return {self.parentPayloadObject:
                    {self.payloadObj: attributes}}
        else:
            return {self.payloadObj: attributes}