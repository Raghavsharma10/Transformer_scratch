def to_json(self):
        """
        put the object to json and remove the internal stuff
        salesking schema stores the type in the title
        """
        data = json.dumps(self)

        out = u'{"%s":%s}' % (self.schema['title'], data)
        return out