def load(self):
        """ Function load
        Get the list of all objects

        @return RETURN: A ForemanItem list
        """
        cl_tmp = self.api.list(self.objName, limit=self.searchLimit).values()
        cl = []
        for i in cl_tmp:
            cl.extend(i)
        return {x[self.index]: ItemPuppetClass(self.api, x['id'],
                                               self.objName, self.payloadObj,
                                               x)
                for x in cl}