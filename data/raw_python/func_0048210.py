def _rngs(self):
      """This is where we should also enforce evidence requirements"""
      outputs = []
      if len(self._exon_groups)==1:
         return [self._exon_groups.consensus('single')]
      z = 0 #output count
      begin = 0
      meeting_criteria = [i for i in range(0,len(self._exon_groups)) if  len(self._exon_groups) >= self._evidence]
      if len(meeting_criteria) == 0: return []
      finish = len(meeting_criteria)
      if len(meeting_criteria) > 0:
         begin = meeting_criteria[0]
         finish = meeting_criteria[-1]
      for i in range(0,len(self._exon_groups)):
         if z == begin:
            outputs.append(self._exon_groups[i].consensus('leftmost'))
         elif z == finish:
            #len(self._exon_groups)-1:
            outputs.append(self._exon_groups[i].consensus('rightmost'))
         else:
            outputs.append(self._exon_groups[i].consensus('internal'))
         z += 1
      v = [seqtools.structure.transcript.Exon(x) for x in outputs]
      v[0].set_leftmost()
      v[-1].set_rightmost()
      return v