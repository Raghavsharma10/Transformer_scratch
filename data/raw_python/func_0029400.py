def analyze(self, scratch, **kwargs):
        """Run and return the results from the SpriteNaming plugin."""
        for sprite in self.iter_sprites(scratch):
            for default in self.default_names:
                if default in sprite.name:
                    self.total_default += 1
                    self.list_default.append(sprite.name)