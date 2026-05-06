def reset(self, seed):
        """
        Reset this generator's seed generator and any clones.
        """
        logger.debug(f'Resetting {self} (seed={seed})')
        self.seed_generator.reset(seed)

        for c in self.clones:
            c.reset(seed)