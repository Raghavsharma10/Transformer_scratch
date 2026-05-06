def generate(self):
        """Runs generation process."""
        for root, _, files in os.walk(self.source_dir):
            for fname in files:
                source_fpath = os.path.join(root, fname)
                self.generate_api_for_source(source_fpath)