def analyze(self, scratch, **kwargs):
        """Run and return the results of the VariableInitialization plugin."""
        variables = dict((x, self.variable_state(x.scripts, x.variables))
                         for x in scratch.sprites)
        variables['global'] = self.variable_state(self.iter_scripts(scratch),
                                                  scratch.stage.variables)
        # Output for now
        import pprint
        pprint.pprint(variables)
        return {'variables': variables}