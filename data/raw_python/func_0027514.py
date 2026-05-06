def _convert_nodelist(self, impl_nodelist):
        """
        Convert a list of underlying implementation nodes into a list of
        *xml4h* wrapper nodes.
        """
        nodelist = [
            self.adapter.wrap_node(n, self.adapter.impl_document, self.adapter)
            for n in impl_nodelist]
        return NodeList(nodelist)