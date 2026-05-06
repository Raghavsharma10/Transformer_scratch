def convert_branch(self, old_node, new_node, comment_dict=None):
        """
        recursively walk a indicator logic tree, starting from a Indicator node.
        converts OpenIOC 1.0 Indicator/IndicatorItems to Openioc 1.1 and preserves order.

        :param old_node: Indicator node, which we walk down to convert
        :param new_node: Indicator node, which we add new IndicatorItem and Indicator nodes too
        :param comment_dict: maps ids to comment values.  only applied to IndicatorItem nodes
        :return: True upon completion
        :raises: UpgradeError if there is a problem during the conversion.
        """
        expected_tag = 'Indicator'
        if old_node.tag != expected_tag:
            raise UpgradeError('old_node expected tag is [%s]' % expected_tag)
        if not comment_dict:
            comment_dict = {}
        for node in old_node.getchildren():
            node_id = node.get('id')
            if node.tag == 'IndicatorItem':
                condition = node.get('condition')
                negation = False
                if condition.endswith('not'):
                    negation = True
                    condition = condition[:-3]
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
                                                              negate=negation,
                                                              nid=node_id)
                # set comment
                comment = node.find('Comment')
                if comment is not None:
                    comment_dict[node_id] = comment.text
                new_node.append(new_ii_node)
            elif node.tag == 'Indicator':
                operator = node.get('operator')
                if operator.upper() not in ['OR', 'AND']:
                    raise UpgradeError('Indicator@operator is not AND/OR. [%s] has [%s]' % (node_id, operator))
                new_i_node = ioc_api.make_indicator_node(operator, node_id)
                new_node.append(new_i_node)
                self.convert_branch(node, new_i_node, comment_dict)
            else:
                # should never get here
                raise UpgradeError('node is not a Indicator/IndicatorItem')
        return True