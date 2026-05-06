def execute(self, parents=None):
        """ Run the data query and construct entities from it's results. """
        results = OrderedDict()
        for row in self.query(parents=parents).execute(self.context.graph):
            data = {k: v.toPython() for (k, v) in row.asdict().items()}
            id = data.get(self.id)
            if id not in results:
                results[id] = self.base_object(data)

            for child in self.children:
                if child.id in data:
                    name = child.get_name(data)
                    value = data.get(child.id)
                    if child.node.many and \
                            child.node.op not in [OP_IN, OP_NIN]:
                        if name not in results[id]:
                            results[id][name] = [value]
                        else:
                            results[id][name].append(value)
                    else:
                        results[id][name] = value
        return results