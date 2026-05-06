def set_attributes(self, **attributes):
        """ Set group of attributes without calling set between attributes regardless of global auto_set.

        Set will be called only after all attributes are set based on global auto_set.

        :param attributes: dictionary of <attribute, value> to set.
        """

        auto_set = IxeObject.get_auto_set()
        IxeObject.set_auto_set(False)
        for name, value in attributes.items():
            setattr(self, name, value)
        if auto_set:
            self.ix_set()
        IxeObject.set_auto_set(auto_set)