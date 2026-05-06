def flatten(self, data=None):
        """reduce all objects into simplified values as a attr dictionary that
        could be transformed back into a full configuration via inflate()"""
        if data == None: data=self.attrs
        ret = {}
        for k,v in iteritems(data):
            if not v: continue # don't flatten if there's nothing to flatten
            elif k == "expo":               v = v.type
            elif k == "version":            v = v.label
            elif k == "ladder":             v = v.name
            elif k == "players":
                newPs = []
                for i,p in enumerate(v):
                    try:    diff = p.difficulty.type
                    except: diff = p.difficulty
                    if isinstance(p, PlayerPreGame):    newPs.append( (p.name, p.type.type, p.initCmd, p.initOptions, diff, p.rating, p.selectedRace.type, self.numObserve, p.playerID, p.raceDefault) )
                    else:                               newPs.append( (p.name, p.type.type, p.initCmd, p.initOptions, diff, p.rating) )
                # TODO -- handle if type or observers params are not available (i.e. if a simple PlayerRecord, not a PlayerPreGame
                ret[k] = newPs
                continue
            elif k == "mode"   and self.mode:   v = v.type
            #elif k == "state":
            elif k == "themap" and self.themap: v = v.name
            ret[k] = v
        return ret