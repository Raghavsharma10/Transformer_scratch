def name(self):
        """ access to classic name attribute is hidden by this property """
        return self.NAME_SEPARATOR.join([super(InfinityVertex, self).name, self.NAME_SUFFIX])