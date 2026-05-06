def options(self, scriptable=None):
        """Return a list of valid options to a menu insert, given a
        Scriptable for context.

        Mostly complete, excepting 'attribute'.

        """
        options = list(Insert.KIND_OPTIONS.get(self.kind, []))
        if scriptable:
            if self.kind == 'var':
                options += scriptable.variables.keys()
                options += scriptable.project.variables.keys()
            elif self.kind == 'list':
                options += scriptable.lists.keys()
                options += scriptable.project.lists.keys()
            elif self.kind == 'costume':
                options += [c.name for c in scriptable.costumes]
            elif self.kind == 'backdrop':
                options += [c.name for c in scriptable.project.stage.costumes]
            elif self.kind == 'sound':
                options += [c.name for c in scriptable.sounds]
                options += [c.name for c in scriptable.project.stage.sounds]
            elif self.kind in ('spriteOnly', 'spriteOrMouse', 'spriteOrStage',
                    'touching'):
                options += [s.name for s in scriptable.project.sprites]
            elif self.kind == 'attribute':
                pass # TODO
            elif self.kind == 'broadcast':
                options += list(set(scriptable.project.get_broadcasts()))
        return options