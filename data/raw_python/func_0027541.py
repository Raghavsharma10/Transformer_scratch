def element(self):
        """
        :return: the :class:`Element` that contains these attributes.
        """
        return self.adapter.wrap_node(
            self.impl_element, self.adapter.impl_document, self.adapter)