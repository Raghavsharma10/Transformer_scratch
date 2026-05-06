def start_evaluating(self, n: Node, s: ShExJ.shapeExpr) -> Optional[bool]:
        """Indicate that we are beginning to evaluate n according to shape expression s.
        If we are already in the process of evaluating (n,s), as indicated self.evaluating, we return our current
        guess as to the result.

        :param n: Node to be evaluated
        :param s: expression for node evaluation
        :return: Assumed evaluation result.  If None, evaluation must be performed
        """
        if not s.id:
            s.id = str(BNode())                 # Random permanant id
        key = (n, s.id)

        # We only evaluate a node once
        if key in self.known_results:
            return self.known_results[key]

        if key not in self.evaluating:
            self.evaluating.add(key)
            return None
        elif key not in self.assumptions:
            self.assumptions[key] = True
        return self.assumptions[key]