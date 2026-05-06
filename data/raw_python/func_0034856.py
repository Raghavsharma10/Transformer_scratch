def get_top(self, stat, n):
        """Return the top n values when sorting by 'stat'"""
        return sorted(self.stats, key=lambda x: getattr(x, stat), reverse=True)[:n]