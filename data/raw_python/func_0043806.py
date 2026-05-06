def agents(self):
        """identify which players are agents (not observers or computers). Errors if flattened."""
        ret = []
        for player in self.players:
            if player.isComputer: continue
            try:
                if player.observer: continue
            except: pass
            ret.append(player)
        return ret