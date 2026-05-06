def _generate_draw_order(self, node=None):
        """Return a list giving the order to draw visuals.
        
        Each node appears twice in the list--(node, True) appears before the
        node's children are drawn, and (node, False) appears after.
        """
        if node is None:
            node = self._scene
        order = [(node, True)]
        children = node.children
        children.sort(key=lambda ch: ch.order)
        for ch in children:
            order.extend(self._generate_draw_order(ch))
        order.append((node, False))
        return order