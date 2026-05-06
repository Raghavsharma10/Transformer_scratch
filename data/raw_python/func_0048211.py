def add_transcripts(self,txs):
      """We traverse through the other transcripts and try to add to these groups
      """
      passed = []
      for tx2 in txs:
         for tx1 in self._initial:
            jov = tx1.junction_overlap(tx2,self._tolerance)
            sub = jov.is_subset()
            if sub == 1 or sub == 2:
               passed.append(tx2)
               break
      if len(passed) == 0:
         sys.stderr.write("Warning unable to add\n")
         return
      for tx in txs:
         self.add_transcript(tx)
      return