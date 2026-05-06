def allLobbySlots(self):
        """the current configuration of the lobby's players, defined before the match starts"""
        if self.debug:
            p      = ["Lobby Configuration detail:"] + \
                     ["    %s:%s%s"%(p, " "*(12-len(p.type)), p.name)]
                     #["    agent:     %s"%p for p in self.agents] + \
                     #["    computer:  %s, %s"%(r,d) for r,d in self.computers]
            if self.observers: # must separate condition because numObs is a number, not an iterator
                p += ["    observers: %d"%self.observers]
            print(os.linesep.join(p))
        return (self.agents, self.computers, self.observers)