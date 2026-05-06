def get_transcripts(self):
    """ a list of the transcripts in the locus"""
    txs = []
    for pays in [x.payload for x in self.g.get_nodes()]:
      for pay in pays:
        txs.append(pay)
    return txs