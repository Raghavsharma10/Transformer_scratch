def parameters(self):
        """ A property that returns all of the model's parameters. """
        parameters = []
        for hl in self.hidden_layers:
            parameters.extend(hl.parameters)
        parameters.extend(self.top_layer.parameters)
        return parameters