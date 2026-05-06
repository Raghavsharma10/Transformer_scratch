def add_transcript(self,tx):
      """add a single transcript"""
      candidates = tx.junctions
      targets = self.junctions
      matches = []
      for i in range(0,len(targets)):
         for j in range(0,len(candidates)):
            if targets[i].overlaps(candidates[j],self._tolerance):
               matches.append([i,j])
      if len(matches) != len(candidates): return
      if len(matches) > 1:
         if False in  [(matches[i+1][0]-matches[i][0])==1 and
                       (matches[i+1][1]-matches[i][1])==1 for i in range(0,len(matches)-1)]:
            return
      # nowe we can add them
      for m in matches:
         self._exon_groups[m[0]].add_exon(tx.exons[m[1]])
         self._exon_groups[m[0]+1].add_exon(tx.exons[m[1]+1])
      self._transcripts.append(tx)