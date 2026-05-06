def get_node_text(self, node, depth, sep, params=False,):
        """
        Get the text for a given Indicator or IndicatorItem node.
        This does walk an IndicatorItem node to get its children text as well.

        :param node: Node to get the text for.
        :param depth: Track the number of recursions that have occured, modifies the indentation.
        :param sep: Seperator used for formatting the text.  Multiplied by the depth to get the indentation.
        :param params: Boolean, set to True in order to display node parameters.
        :return:
        """
        indent = sep * depth
        s = ''
        tag = node.tag
        if tag == 'Indicator':
            node_text = self.get_i_text(node)
        elif tag == 'IndicatorItem':
            node_text = self.get_ii_text(node)
        else:
            raise IOCParseError('Invalid node encountered: {}'.format(tag))
        s += '{}{}\n'.format(indent, node_text)
        if params:
            param_text = self.get_param_text(node.attrib.get('id'))
            for pt in param_text:
                s += '{}{}\n'.format(indent+sep, pt)
        if node.tag == 'Indicator':
            for child in node.getchildren():
                s += self.get_node_text(node=child, depth=depth+1, sep=sep, params=params)
        return s