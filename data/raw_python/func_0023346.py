def _visual_bounds_at(self, pos, node=None):
        """Find a visual whose bounding rect encompasses *pos*.
        """
        if node is None:
            node = self.scene
            
        for ch in node.children:
            hit = self._visual_bounds_at(pos, ch)
            if hit is not None:
                return hit
        
        if (not isinstance(node, VisualNode) or not node.visible or 
                not node.interactive):
            return None
        
        bounds = [node.bounds(axis=i) for i in range(2)]
        
        if None in bounds:
            return None
        
        tr = self.scene.node_transform(node).inverse
        corners = np.array([
            [bounds[0][0], bounds[1][0]],
            [bounds[0][0], bounds[1][1]],
            [bounds[0][1], bounds[1][0]],
            [bounds[0][1], bounds[1][1]]])
        bounds = tr.map(corners)
        xhit = bounds[:, 0].min() < pos[0] < bounds[:, 0].max()
        yhit = bounds[:, 1].min() < pos[1] < bounds[:, 1].max()
        if xhit and yhit:
            return node