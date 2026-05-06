def load(self):
        """ Function load
        Get the list of all objects

        @return RETURN: A ForemanItem list
        """
        return {x[self.index]: self.itemType(self.api, x['id'],
                                             self.objName, self.payloadObj,
                                             x)
                for x in self.api.list(self.objName,
                                       limit=self.searchLimit)}