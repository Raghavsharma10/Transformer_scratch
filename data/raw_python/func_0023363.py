def get_free_region(self, width, height):
        """Get a free region of given size and allocate it

        Parameters
        ----------
        width : int
            Width of region to allocate
        height : int
            Height of region to allocate

        Returns
        -------
        bounds : tuple | None
            A newly allocated region as (x, y, w, h) or None
            (if failed).
        """
        best_height = best_width = np.inf
        best_index = -1
        for i in range(len(self._atlas_nodes)):
            y = self._fit(i, width, height)
            if y >= 0:
                node = self._atlas_nodes[i]
                if (y+height < best_height or
                        (y+height == best_height and node[2] < best_width)):
                    best_height = y+height
                    best_index = i
                    best_width = node[2]
                    region = node[0], y, width, height
        if best_index == -1:
            return None

        node = region[0], region[1] + height, width
        self._atlas_nodes.insert(best_index, node)
        i = best_index+1
        while i < len(self._atlas_nodes):
            node = self._atlas_nodes[i]
            prev_node = self._atlas_nodes[i-1]
            if node[0] < prev_node[0]+prev_node[2]:
                shrink = prev_node[0]+prev_node[2] - node[0]
                x, y, w = self._atlas_nodes[i]
                self._atlas_nodes[i] = x+shrink, y, w-shrink
                if self._atlas_nodes[i][2] <= 0:
                    del self._atlas_nodes[i]
                    i -= 1
                else:
                    break
            else:
                break
            i += 1

        # Merge nodes
        i = 0
        while i < len(self._atlas_nodes)-1:
            node = self._atlas_nodes[i]
            next_node = self._atlas_nodes[i+1]
            if node[1] == next_node[1]:
                self._atlas_nodes[i] = node[0], node[1], node[2]+next_node[2]
                del self._atlas_nodes[i+1]
            else:
                i += 1

        return region