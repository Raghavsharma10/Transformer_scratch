def get_prediction(self, features=None, tag=None, namespaces=None):
        """Send an unlabeled example to the trained VW instance.
        Uses any given features or namespaces, as well as any previously
        added namespaces (using them up in the process).

        Returns a VWResult object."""
        if features is not None:
            namespace = Namespace(features=features)
            self.add_namespace(namespace)
        result = self.send_example(tag=tag, namespaces=namespaces)
        return result