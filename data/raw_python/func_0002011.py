def criteria_text(self, sep='  ', params=False):
        """
        Get a text representation of the criteria node.

        :param sep: Separator used to indent the contents of the node.
        :param params: Boolean, set to True in order to display node parameters.
        :return:
        """

        s = ''
        criteria_node = self.root.find('criteria')
        if criteria_node is None:
            return s
        node_texts = []
        for node in criteria_node.getchildren():
            nt = self.get_node_text(node, depth=0, sep=sep, params=params)
            node_texts.append(nt)
        s = '\n'.join(node_texts)
        return s