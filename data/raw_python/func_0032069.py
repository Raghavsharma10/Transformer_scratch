def extractValue(self, model, item):
        """
        Get the class name of the factory referenced by a port.

        @param model: Either a TabularDataModel or a ScrollableView, depending
        on what this column is part of.

        @param item: A port item instance (as defined by L{xmantissa.port}).

        @rtype: C{unicode}
        @return: The name of the class of the item to which this column's
        attribute refers.
        """
        factory = super(FactoryColumn, self).extractValue(model, item)
        return factory.__class__.__name__.decode('ascii')