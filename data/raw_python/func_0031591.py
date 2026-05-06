def find_record_files(self):
        """
        Yield paths to record files.
        """
        for (root, _, files) in os.walk(self.record_path):
            for f in (f for f in files if f.endswith('.json')):
                yield os.path.join(root, f)