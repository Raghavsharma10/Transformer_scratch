def getPayloadStruct(self, payload):
        """ Function getPayloadStruct

        @param payload: The payload structure to the object to add
        @return RETURN: A dict
        """
        newSubItem = self.objType(self.api, 0, self.parentObjName,
                                  self.parentPayloadObj, self.parentKey, {})
        return newSubItem.getPayloadStruct(payload, self.parentPayloadObj)