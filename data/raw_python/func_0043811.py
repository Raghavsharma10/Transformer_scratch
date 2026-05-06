def observers(self):
        """the players who are actually observers"""
        ret = []
        for player in self.players:
            try:
                if player.observer: ret.append(player)
            except: pass # ignore PlayerRecords which don't have an observer attribute
        return ret