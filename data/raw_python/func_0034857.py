def save_pstat(self, path):
        """
        Save the modified pstats file
        """
        stats = {}
        for s in self.stats:
            if not s.exclude:
                stats.update(s.to_dict())

        with open(path, 'wb') as f:
            marshal.dump(stats, f)