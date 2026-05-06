def update_from_stats(self, stats):
        """Update columns based on partition statistics"""

        sd = dict(stats)

        for c in self.columns:

            if c not in sd:
                continue

            stat = sd[c]

            if stat.size and stat.size > c.size:
                c.size = stat.size

            c.lom = stat.lom