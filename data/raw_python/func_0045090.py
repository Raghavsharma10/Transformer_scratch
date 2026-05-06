def _get_model_nodes(self, model):
        """
        Find all the non-auto created nodes of the model.
        """
        nodes = [(name, node) for name, node in model._nodes.items()
                if node._is_auto_created is False]
        nodes.sort(key=lambda n: n[0])
        return nodes