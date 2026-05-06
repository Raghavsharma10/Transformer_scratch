def reset(self) -> None:
        """
        Reset the context preceeding an evaluation
        """
        self.evaluating = set()
        self.assumptions = {}
        self.known_results = {}
        self.current_node = None
        self.evaluate_stack = []
        self.bnode_map = {}