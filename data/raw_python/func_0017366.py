def assign_vertices(self):
        """
        Sets .edges, .verts for node positions. 
        X and Y positions here refer to base assumption that tree is right
        facing, reorient_coordinates() will handle re-translating this.        
        """
        # shortname 
        uselen = bool(self.ttree.style.use_edge_lengths)

        # postorder: children then parents (nidxs from 0 up)
        # store edge array for connecting child nodes to parent nodes
        nidx = 0
        for node in self.ttree.treenode.traverse("postorder"):            
            if not node.is_root():
                self.edges[nidx, :] = [node.up.idx, node.idx]
                nidx += 1

        # store verts array with x,y positions of nodes (lengths of branches)
        # we want tips to align at the right face (larger axis number)
        _root = self.ttree.treenode.get_tree_root()
        _treeheight = _root.get_distance(_root.get_farthest_leaf()[0])

        # set node x, y
        tidx = len(self.ttree) - 1
        for node in self.ttree.treenode.traverse("postorder"):

            # Just leaves: x positions are evenly spread and ordered on axis
            if node.is_leaf() and (not node.is_root()):
                
                # set y-positions (heights). Distance from root or zero
                node.y = _treeheight - _root.get_distance(node)
                if not uselen:
                    node.y = 0.0
                
                # set x-positions (order of samples)
                if self.ttree._fixed_order:
                    node.x = self.ttree._fixed_order.index(node.name)# - tidx
                else:
                    node.x = tidx
                    tidx -= 1
                
                # store the x,y vertex positions
                self.verts[node.idx] = [node.x, node.y]

            # All internal node positions are not evenly spread or ordered
            else:
                # height is either distance or nnodes from root
                node.y = _treeheight - _root.get_distance(node)
                if not uselen:
                    node.y = max([i.y for i in node.children]) + 1

                # x position is halfway between childrens x-positions
                if node.children:
                    nch = node.children
                    node.x = sum(i.x for i in nch) / float(len(nch))
                else:
                    node.x = tidx

                # store the x,y vertex positions                    
                self.verts[node.idx] = [node.x, node.y]