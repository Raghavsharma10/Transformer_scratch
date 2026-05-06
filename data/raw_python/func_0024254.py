def __dfs(self, start, weights, depth_limit):
        """
        modified NX dfs
        """
        adj = self._adj

        stack = [(start, depth_limit, iter(sorted(adj[start], key=weights)))]
        visited = {start}
        disconnected = defaultdict(list)
        edges = defaultdict(list)

        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
            except StopIteration:
                stack.pop()
            else:
                if child not in visited:
                    edges[parent].append(child)
                    visited.add(child)
                    if depth_now > 1:
                        front = adj[child].keys() - {parent}
                        if front:
                            stack.append((child, depth_now - 1, iter(sorted(front, key=weights))))
                elif child not in disconnected:
                    disconnected[parent].append(child)

        return visited, edges, disconnected