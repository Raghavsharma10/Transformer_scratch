def getGradient(self,j):
        """ get rotated gradient for fixed effect i """
        i = int(self.indicator['term'][j])
        r = int(self.indicator['row'][j])
        c = int(self.indicator['col'][j])
        rv = -np.kron(self.Fstar()[i][:,[r]],self.Astar()[i][[c],:])
        return rv