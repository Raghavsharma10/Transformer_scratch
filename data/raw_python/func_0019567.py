def average_motifs(self, other, pos, orientation, include_bg=False):
        """Return the average of two motifs.

        Combine this motif with another motif and return the average as a new
        Motif object. The position and orientatien need to be supplied. The pos
        parameter is the position of the second motif relative to this motif.
        
        For example, take the following two motifs:
        Motif 1: CATGYT
        Motif 2: GGCTTGY

        With position -2, the motifs are averaged as follows:
        xxCATGYT
        GGCTTGYx

        Parameters
        ----------
        other : Motif object
            Other Motif object.
        pos : int
            Position of the second motif relative to this motif.
        orientation : int
            Orientation, should be 1 or -1. If the orientation is -1 then the 
            reverse complement of the other motif is used for averaging.
        include_bg : bool , optional
            Extend both motifs with background frequencies (0.25) before
            averaging. False by default.
        
        Returns
        -------
        motif : motif object
            New Motif object containing average motif.
        """
        # xxCATGYT
        # GGCTTGYx
        # pos = -2
        pfm1 = self.pfm[:]
        pfm2 = other.pfm[:]

        if orientation < 0:
            pfm2 = [row[::-1] for row in pfm2[::-1]]
        
        pfm1_count = float(np.sum(pfm1[0]))
        pfm2_count = float(np.sum(pfm2[0]))
        
        if include_bg:
            if len(pfm1) > len(pfm2) + pos:
                pfm2 += [[pfm2_count / 4.0 for x in range(4)] for i in range(-(len(pfm1) - len(pfm2) - pos), 0)]
            elif len(pfm2) + pos > len(pfm1):
                pfm1 += [[pfm1_count / 4.0 for x in range(4)] for i in range(-(len(pfm2) - len(pfm1) + pos), 0)]
        
            if pos < 0:
                pfm1 = [[pfm1_count / 4.0 for x in range(4)] for i in range(-pos)] + pfm1
            elif pos > 0:
                pfm2 = [[pfm2_count / 4.0 for x in range(4)] for i in range(pos)] + pfm2
        
        else:
            if len(pfm1) > len(pfm2) + pos:
                pfm2 += [[pfm1[i][x] / pfm1_count * (pfm2_count) for x in range(4)] for i in range(-(len(pfm1) - len(pfm2) - pos), 0)]
            elif len(pfm2) + pos > len(pfm1):
                pfm1 += [[pfm2[i][x] / pfm2_count * (pfm1_count) for x in range(4)] for i in range(-(len(pfm2) - len(pfm1) + pos), 0)]
        
            if pos < 0:
                pfm1 = [[pfm2[i][x] / pfm2_count * (pfm1_count) for x in range(4)] for i in range(-pos)] + pfm1
            elif pos > 0:
                pfm2 = [[pfm1[i][x] / pfm1_count * (pfm2_count) for x in range(4)] for i in range(pos)] + pfm2

        pfm = [[a + b for a,b in zip(x,y)] for x,y in zip(pfm1, pfm2)]
        
        m = Motif(pfm)
        m.id = m.to_consensus()
        return m