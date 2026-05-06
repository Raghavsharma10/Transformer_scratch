def fromBox(self, name, strings, objects, proto):
        """
        Retreive an attribute from the C{proto} parameter.
        """
        objects[name] = getattr(proto, self.attr)