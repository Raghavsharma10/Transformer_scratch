def analyze(self, scratch, **kwargs):
        """Run and return the results from the Animation plugin."""
        results = Counter()
        for script in self.iter_scripts(scratch):
            gen = self.iter_blocks(script.blocks)
            name = 'start'
            level = None
            while name != '':
                if name in self.ANIMATION:
                    gen, count = self._check_animation(name, level, gen)
                    results.update(count)
                name, level, _ = next(gen, ('', 0, ''))
        return {'animation': results}