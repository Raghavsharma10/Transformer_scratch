def analyze(self, scratch, **kwargs):
        """Run and return the results of the AttributeInitialization plugin."""
        changes = dict((x.name, self.sprite_changes(x)) for x in
                       scratch.sprites)
        changes['stage'] = {
            'background': self.attribute_state(scratch.stage.scripts,
                                               'costume')}
        # self.output_results(changes)
        return {'initialized': changes}