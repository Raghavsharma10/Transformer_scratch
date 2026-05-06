def analyze(self, scratch, **kwargs):
        """Run and return the results from the BlockCounts plugin."""
        file_blocks = Counter()
        for script in self.iter_scripts(scratch):
            for name, _, _ in self.iter_blocks(script.blocks):
                file_blocks[name] += 1
        self.blocks.update(file_blocks)  # Update the overall count
        return {'types': file_blocks}