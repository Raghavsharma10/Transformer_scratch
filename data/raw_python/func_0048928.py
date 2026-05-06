def sort_transcripts(self):
     """Sort the transcripts stored here"""
     txs = sorted(self.transcripts,key=lambda x: (x.range.chr, x.range.start, x.range.end))
     self._transcripts = txs