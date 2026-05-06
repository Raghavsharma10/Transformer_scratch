def node_slider(self, seed=None):
        """
        Returns a toytree copy with node heights modified while retaining 
        the same topology but not necessarily node branching order. 
        Node heights are moved up or down uniformly between their parent 
        and highest child node heights in 'levelorder' from root to tips.
        The total tree height is retained at 1.0, only relative edge
        lengths change.
        """
        # I don't think user's should need to access prop
        prop = 0.999
        assert isinstance(prop, float), "prop must be a float"
        assert prop < 1, "prop must be a proportion >0 and < 1."
        random.seed(seed)

        ctree = self._ttree.copy()
        for node in ctree.treenode.traverse():

            ## slide internal nodes 
            if node.up and node.children:

                ## get min and max slides
                minjit = max([i.dist for i in node.children]) * prop
                maxjit = (node.up.height * prop) - node.height
                newheight = random.uniform(-minjit, maxjit)

                ## slide children
                for child in node.children:
                    child.dist += newheight

                ## slide self to match
                node.dist -= newheight
        ctree._coords.update()
        return ctree