def get_common_ancestor(self, *target_nodes, **kargs):
        """
        Returns the first common ancestor between this node and a given
        list of 'target_nodes'.

        **Examples:**
          t = tree.Tree("(((A:0.1, B:0.01):0.001, C:0.0001):1.0[&&NHX:name=common], (D:0.00001):0.000001):2.0[&&NHX:name=root];")
          A = t.get_descendants_by_name("A")[0]
          C = t.get_descendants_by_name("C")[0]
          common =  A.get_common_ancestor(C)
          print common.name
        """
        get_path = kargs.get("get_path", False)

        if len(target_nodes) == 1 and type(target_nodes[0]) \
                in set([set, tuple, list, frozenset]):
            target_nodes = target_nodes[0]

        # Convert node names into node instances
        target_nodes = _translate_nodes(self, *target_nodes)

        # If only one node is provided, use self as the second target
        if type(target_nodes) != list:
            target_nodes = [target_nodes, self]

        n2path = {}
        reference = []
        ref_node = None
        for n in target_nodes:
            current = n
            while current:
                n2path.setdefault(n, set()).add(current)
                if not ref_node:
                    reference.append(current)
                current = current.up
            if not ref_node:
                ref_node = n

        common = None
        for n in reference:
            broken = False
            for node, path in six.iteritems(n2path):
                if node is not ref_node and n not in path:
                    broken = True
                    break

            if not broken:
                common = n
                break
        if not common:
            raise TreeError("Nodes are not connected!")

        if get_path:
            return common, n2path
        else:
            return common