def fix_schema_node_ordering(parent):
        """
        Fix the ordering of children under the criteria node to ensure that IndicatorItem/Indicator order
         is preserved, as per XML Schema.
        :return:
        """
        children = parent.getchildren()
        i_nodes = [node for node in children if node.tag == 'IndicatorItem']
        ii_nodes = [node for node in children if node.tag == 'Indicator']
        if not ii_nodes:
            return
        # Remove all the children
        for node in children:
            parent.remove(node)
        # Add the Indicator nodes back
        for node in i_nodes:
            parent.append(node)
        # Now add the IndicatorItem nodes back
        for node in ii_nodes:
            parent.append(node)
        # Now recurse
        for node in ii_nodes:
            fix_schema_node_ordering(node)