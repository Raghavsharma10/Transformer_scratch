def remove_transcript(self,tx_id):
    """Remove a transcript from the locus by its id

    :param tx_id:
    :type tx_id: string
    """
    txs = self.get_transcripts()
    if tx_id not in [x.id for x in txs]:
      return
    tx = [x for x in txs if x.id==tx_id][0]
    for n in [x for x in self.g.get_nodes()]:
      if tx_id not in [y.id for y in n.payload]:
        continue
      n.payload.remove(tx)
      if len(n.payload)==0:
        self.g.remove_node(n)