def resolve_polytomy(self, 
        default_dist=0.0, 
        default_support=0.0,
        recursive=True):
        """
        Resolve all polytomies under current node by creating an
        arbitrary dicotomic structure among the affected nodes. This
        function randomly modifies current tree topology and should
        only be used for compatibility reasons (i.e. programs
        rejecting multifurcated node in the newick representation).

        :param 0.0 default_dist: artificial branch distance of new
            nodes.

        :param 0.0 default_support: artificial branch support of new
            nodes.

        :param True recursive: Resolve any polytomy under this
             node. When False, only current node will be checked and fixed.
        """
        def _resolve(node):
            if len(node.children) > 2:
                children = list(node.children)
                node.children = []
                next_node = root = node
                for i in range(len(children) - 2):
                    next_node = next_node.add_child()
                    next_node.dist = default_dist
                    next_node.support = default_support

                next_node = root
                for ch in children:
                    next_node.add_child(ch)
                    if ch != children[-2]:
                        next_node = next_node.children[0]
        target = [self]
        if recursive:
            target.extend([n for n in self.get_descendants()])
        for n in target:
            _resolve(n)