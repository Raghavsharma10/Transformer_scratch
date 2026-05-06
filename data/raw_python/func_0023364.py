def _fit(self, index, width, height):
        """Test if region (width, height) fit into self._atlas_nodes[index]"""
        node = self._atlas_nodes[index]
        x, y = node[0], node[1]
        width_left = width
        if x+width > self._shape[1]:
            return -1
        i = index
        while width_left > 0:
            node = self._atlas_nodes[i]
            y = max(y, node[1])
            if y+height > self._shape[0]:
                return -1
            width_left -= node[2]
            i += 1
        return y