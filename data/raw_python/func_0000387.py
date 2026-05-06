def append(self, payload):
        """ Function __iadd__

        @param payload: The payload corresponding to the object to add
        @return RETURN: A ForemanItem
        """
        if self.objType.setInParentPayload:
            print('Error, {} is not elibible to addition, but only set'
                  .format(self.objName))
            return False
        ret = self.api.create("{}/{}/{}".format(self.parentObjName,
                                                self.parentKey,
                                                self.objNameSet),
                              self.getPayloadStruct(payload))
        return ret