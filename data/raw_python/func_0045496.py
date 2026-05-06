def nextProperty(self, propuri):
        """Returns the next property in the list of properties. If it's the last one, returns the first one."""
        if propuri == self.properties[-1].uri:
            return self.properties[0]
        flag = False
        for x in self.properties:
            if flag == True:
                return x
            if x.uri == propuri:
                flag = True
        return None