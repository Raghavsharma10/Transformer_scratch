def whoAmI(self):
        """return the player object that owns this configuration"""
        self.inflate() # ensure self.players contains player objects
        if self.thePlayer:
            for p in self.players:
                if p.name != self.thePlayer: continue
                return p
        elif len(self.players) == 1:
            ret = self.players[0]
            self.thePlayer = ret.name # remember this for the future in case more players are added
            return ret
        raise Exception("could not identify which player this is given %s (%s)"%(self.players, self.thePlayer))