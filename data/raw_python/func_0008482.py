def find_keywords(self, string, **kwargs):
        """ Returns a sorted list of keywords in the given string.
        """
        return find_keywords(string,
                     parser = self,
                        top = kwargs.pop("top", 10),
                  frequency = kwargs.pop("frequency", {}), **kwargs
        )