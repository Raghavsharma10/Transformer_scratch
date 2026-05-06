def attributes(self):
        """
        Get or set this element's attributes as name/value pairs.

        .. note::
            Setting element attributes via this accessor will **remove**
            any existing attributes, as opposed to the :meth:`set_attributes`
            method which only updates and replaces them.
        """
        attr_impl_nodes = self.adapter.get_node_attributes(self.impl_node)
        return AttributeDict(attr_impl_nodes, self.impl_node, self.adapter)