def _objectify(self, node, binding, depth, path):
        """ Given an RDF node URI (and it's associated schema), return an
        object from the ``graph`` that represents the information available
        about this node. """
        if binding.is_object:
            obj = {'$schema': binding.path}
            for (s, p, o) in self.graph.triples((node, None, None)):
                prop = binding.get_property(p)
                if prop is None or depth <= 1 or o in path:
                    continue
                # This is slightly odd but yield purty objects:
                if depth <= 2 and (prop.is_array or prop.is_object):
                    continue
                sub_path = path.union([node])
                value = self._objectify(o, prop, depth - 1, sub_path)
                if prop.is_array and prop.name in obj:
                    obj[prop.name].extend(value)
                else:
                    obj[prop.name] = value
            return obj
        elif binding.is_array:
            for item in binding.items:
                return [self._objectify(node, item, depth, path)]
        else:
            return node.toPython()