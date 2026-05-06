def pyxwriter(self):
        """Update the pyx file."""
        model = self.Model()
        if hasattr(self, 'Parameters'):
            model.parameters = self.Parameters(vars(self))
        else:
            model.parameters = parametertools.Parameters(vars(self))
        if hasattr(self, 'Sequences'):
            model.sequences = self.Sequences(model=model, **vars(self))
        else:
            model.sequences = sequencetools.Sequences(model=model,
                                                      **vars(self))
        return PyxWriter(self, model, self.pyxfilepath)