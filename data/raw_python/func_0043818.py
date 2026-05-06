def toJson(self, data=None, pretty=False):
        """convert the flattened dictionary into json"""
        if data==None: data = self.attrs
        data = self.flatten(data) # don't send objects as str in json
        #if pretty:
        ret = json.dumps(data, indent=4, sort_keys=True)
        #self.inflate() # restore objects from json str data
        return ret