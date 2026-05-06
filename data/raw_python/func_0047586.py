def only_path(self):
        """Finds the only path from the start node. If there is more than one,
        raises ValueError."""
        start = [v for v in self.nodes if self.nodes[v].get('start', False)]
        if len(start) != 1: 
            raise ValueError("graph does not have exactly one start node")

        path = []
        [v] = start
        while True:
            path.append(v)
            u = v
            vs = self.edges.get(u, ())
            if len(vs) == 0:
                break
            elif len(vs) > 1:
                raise ValueError("graph does not have exactly one path")
            [v] = vs

        return path