def parents(self, term, recursive=False, **kwargs):
        """ Returns a list of all semantic types for the given term.
            If recursive=True, traverses parents up to the root.
        """
        def dfs(term, recursive=False, visited={}, **kwargs):
            if term in visited: # Break on cyclic relations.
                return []
            visited[term], a = True, []
            if dict.__contains__(self, term):
                a = self[term][0].keys()
            for classifier in self.classifiers:
                a.extend(classifier.parents(term, **kwargs) or [])
            if recursive:
                for w in a: a += dfs(w, recursive, visited, **kwargs)
            return a
        return unique(dfs(self._normalize(term), recursive, {}, **kwargs))