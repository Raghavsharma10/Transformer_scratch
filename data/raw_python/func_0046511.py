def partition_loci(self,verbose=False):
    """ break the locus up into unconnected loci

    :return: list of loci
    :rtype: TranscriptLoci[]
    """
    self.g.merge_cycles()
    #sys.stderr.write(self.g.get_report()+"\n")
    gs = self.g.partition_graph(verbose=verbose)
    tls = [] # makea list of transcript loci
    for g in gs:
      tl = TranscriptLoci()
      tl.merge_rules = self.merge_rules
      ns = g.get_nodes()
      for n in [x.payload for x in ns]:
        for tx in n:
          tl.add_transcript(tx)
      if len(tl.g.get_nodes()) > 0:
        tls.append(tl)
    #print '-----------------------' 
    #names = []
    #for tl in tls:
    #  for tx in tl.get_transcripts():
    #    names.append(tx.get_gene_name())
    #for name in sorted(names):
    #  print name
    #print '--------------------------'
    return tls