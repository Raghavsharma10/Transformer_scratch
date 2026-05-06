def collect(self, parents=None):
        """ Given re-constructed entities, conduct queries for child
        entities and merge them into the current level's object graph. """
        results = self.execute(parents=parents)
        ids = results.keys()
        for child in self.nested():
            name = child.node.name
            for child_data in child.collect(parents=ids).values():
                parent_id = child_data.pop('$parent', None)
                if child.node.many:
                    if name not in results[parent_id]:
                        results[parent_id][name] = []
                    results[parent_id][name].append(child_data)
                else:
                    results[parent_id][name] = child_data
        return results