def enter_node(self, ir_node):
        """
        Enter the given element; keeps track of `cdata`;
        subclasses may extend by overriding
        """
        this_is_cdata = (isinstance(ir_node, Element)
                         and ir_node.name in self.cdata_elements)
        self.state['is_cdata'] = bool(self.state.get('is_cdata')) or this_is_cdata