def add_stats(self, args):
        """Callback to add motif statistics."""
        bg_name, stats = args
        logger.debug("Stats: %s %s", bg_name, stats)
        
        for motif_id in stats.keys():
            if motif_id not in self.stats:
                self.stats[motif_id] = {}
        
            self.stats[motif_id][bg_name] = stats[motif_id]