def copy(self):
        """Return a new Project instance, deep-copying all the attributes."""
        p = Project()
        p.name = self.name
        p.path = self.path
        p._plugin = self._plugin
        p.stage = self.stage.copy()
        p.stage.project = p

        for sprite in self.sprites:
            s = sprite.copy()
            s.project = p
            p.sprites.append(s)

        for actor in self.actors:
            if isinstance(actor, Sprite):
                p.actors.append(p.get_sprite(actor.name))
            else:
                a = actor.copy()
                if isinstance(a, Watcher):
                    if isinstance(a.target, Project):
                        a.target = p
                    elif isinstance(a.target, Stage):
                        a.target = p.stage
                    else:
                        a.target = p.get_sprite(a.target.name)
                p.actors.append(a)

        p.variables = dict((n, v.copy()) for (n, v) in self.variables.items())
        p.lists = dict((n, l.copy()) for (n, l) in self.lists.items())
        p.thumbnail = self.thumbnail
        p.tempo = self.tempo
        p.notes = self.notes
        p.author = self.author
        return p