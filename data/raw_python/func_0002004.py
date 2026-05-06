def remove_indicator(self, nid, prune=False):
        """
        Removes a Indicator or IndicatorItem node from the IOC.  By default,
        if nodes are removed, any children nodes are inherited by the removed
        node. It has the  ability to delete all children Indicator and
        IndicatorItem nodes underneath an Indicator node if the 'prune'
        argument is set.

        This will not remove the top level Indicator node from an IOC.
        If the id value has been reused within the IOC, this will remove the
        first node which contains the id value.

        This also removes any parameters associated with any nodes that are
        removed.

        :param nid: The Indicator/@id or IndicatorItem/@id value indicating a specific node to remove.
        :param prune: Remove all children of the deleted node. If a Indicator node is removed and prune is set to
         False, the children nodes will be promoted to be children of the removed nodes' parent.
        :return: True if nodes are removed, False otherwise.
        """
        try:
            node_to_remove = self.top_level_indicator.xpath(
                '//IndicatorItem[@id="{}"]|//Indicator[@id="{}"]'.format(str(nid), str(nid)))[0]
        except IndexError:
            log.exception('Node [{}] not present'.format(nid))
            return False
        if node_to_remove.tag == 'IndicatorItem':
            node_to_remove.getparent().remove(node_to_remove)
            self.remove_parameter(ref_id=nid)
            return True
        elif node_to_remove.tag == 'Indicator':
            if node_to_remove == self.top_level_indicator:
                raise IOCParseError('Cannot remove the top level indicator')
            if prune:
                pruned_ids = node_to_remove.xpath('.//@id')
                node_to_remove.getparent().remove(node_to_remove)
                for pruned_id in pruned_ids:
                    self.remove_parameter(ref_id=pruned_id)
            else:
                for child_node in node_to_remove.getchildren():
                    node_to_remove.getparent().append(child_node)
                node_to_remove.getparent().remove(node_to_remove)
                self.remove_parameter(ref_id=nid)
            return True
        else:
            raise IOCParseError(
                'Bad tag found.  Expected "IndicatorItem" or "Indicator", got [[}]'.format(node_to_remove.tag))