def load(self, limit=9999):
        """ Function list
        Get the list of all interfaces

        @param key: The targeted object
        @param limit: The limit of items to return
        @return RETURN: A ForemanItem list
        """
        subItemList = self.api.list('{}/{}/{}'.format(self.parentObjName,
                                                      self.parentKey,
                                                      self.objName,
                                                      ),
                                    limit=limit)
        if self.objName == 'puppetclass_ids':
            subItemList = list(map(lambda x: {'id': x}, subItemList))
        if self.objName == 'puppetclasses':
            sil_tmp = subItemList.values()
            subItemList = []
            for i in sil_tmp:
                subItemList.extend(i)
        return {x[self.index]: self.objType(self.api, x['id'],
                                            self.parentObjName,
                                            self.parentPayloadObj,
                                            self.parentKey,
                                            x)
                for x in subItemList}