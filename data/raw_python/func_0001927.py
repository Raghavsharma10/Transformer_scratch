def convert_branch(self, old_node, new_node, ids_to_skip, comment_dict=None):
        """
        Recursively walk a indicator logic tree, starting from a Indicator node.
        Converts OpenIOC 1.1 Indicator/IndicatorItems to Openioc 1.0 and preserves order.


        :param old_node: An Indicator node, which we walk down to convert
        :param new_node: An Indicator node, which we add new IndicatorItem and Indicator nodes too
        :param ids_to_skip: set of node @id values not to convert
        :param comment_dict: maps ids to comment values.  only applied to IndicatorItem nodes
        :return: returns True upon completion.
        :raises: DowngradeError if there is a problem during the conversion.
        """
        expected_tag = 'Indicator'
        if old_node.tag != expected_tag:
            raise DowngradeError('old_node expected tag is [%s]' % expected_tag)
        if not comment_dict:
            comment_dict = {}
        for node in old_node.getchildren():
            node_id = node.get('id')
            if node_id in ids_to_skip:
                continue
            if node.tag == 'IndicatorItem':
                negation = node.get('negate')
                condition = node.get('condition')
                if 'true' in negation.lower():
                    new_condition = condition + 'not'
                else:
                    new_condition = condition
                document = node.xpath('Context/@document')[0]
                search = node.xpath('Context/@search')[0]
                content_type = node.xpath('Content/@type')[0]
                content = node.findtext('Content')
                context_type = node.xpath('Context/@type')[0]
                new_ii_node = ioc_api.make_indicatoritem_node(condition=condition,
                                                              document=document,
                                                              search=search,
                                                              content_type=content_type,
                                                              content=content,
                                                              context_type=context_type,
                                                              nid=node_id)
                # set condition
                new_ii_node.attrib['condition'] = new_condition
                # set comment
                if node_id in comment_dict:
                    comment = comment_dict[node_id]
                    comment_node = et.Element('Comment')
                    comment_node.text = comment
                    new_ii_node.append(comment_node)
                # remove preserver-case and negate
                del new_ii_node.attrib['negate']
                del new_ii_node.attrib['preserve-case']
                new_node.append(new_ii_node)
            elif node.tag == 'Indicator':
                operator = node.get('operator')
                if operator.upper() not in ['OR', 'AND']:
                    raise DowngradeError('Indicator@operator is not AND/OR. [%s] has [%s]' % (node_id, operator))
                new_i_node = ioc_api.make_indicator_node(operator, node_id)
                new_node.append(new_i_node)
                self.convert_branch(node, new_i_node, ids_to_skip, comment_dict)
            else:
                # should never get here
                raise DowngradeError('node is not a Indicator/IndicatorItem')
        return True