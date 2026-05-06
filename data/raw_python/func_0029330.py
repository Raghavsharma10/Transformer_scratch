def analyze(self, scratch, **kwargs):
        """Run and return the results form the DeadCode plugin.

        The variable_event indicates that the Scratch file contains at least
        one instance of a broadcast event based on a variable. When
        variable_event is True, dead code scripts reported by this plugin that
        begin with a "when I receive" block may not actually indicate dead
        code.

        """
        self.total_instances += 1
        sprites = {}
        for sprite, script in self.iter_sprite_scripts(scratch):
            if not script.reachable:
                sprites.setdefault(sprite, []).append(script)
        if sprites:
            self.dead_code_instances += 1
            import pprint
            pprint.pprint(sprites)
        variable_event = any(True in self.get_broadcast_events(x) for x in
                             self.iter_scripts(scratch))
        return {'dead_code': {'sprites': sprites,
                              'variable_event': variable_event}}