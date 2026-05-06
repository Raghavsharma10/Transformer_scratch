def delete(self, prevent_nondicotomic=True, preserve_branch_length=False):
        """
        Deletes node from the tree structure. Notice that this method
        makes 'disappear' the node from the tree structure. This means
        that children from the deleted node are transferred to the
        next available parent.

        Parameters:
        -----------
        prevent_nondicotomic: 
            When True (default), delete
            function will be execute recursively to prevent single-child
            nodes.

        preserve_branch_length: 
            If True, branch lengths of the deleted nodes are transferred 
            (summed up) to its parent's branch, thus keeping original 
            distances among nodes.

        **Example:**
                / C
          root-|
               |        / B
                \--- H |
                        \ A

          > H.delete() will produce this structure:

                / C
               |
          root-|--B
               |
                \ A

        """
        parent = self.up
        if parent:
            if preserve_branch_length:
                if len(self.children) == 1:
                    self.children[0].dist += self.dist
                elif len(self.children) > 1:
                    parent.dist += self.dist

            for ch in self.children:
                parent.add_child(ch)

            parent.remove_child(self)

        # Avoids parents with only one child
        if prevent_nondicotomic and parent and\
              len(parent.children) < 2:
            parent.delete(prevent_nondicotomic=False,
                          preserve_branch_length=preserve_branch_length)