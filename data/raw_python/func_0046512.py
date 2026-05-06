def add_transcript(self,tx):
    """Add a transcript to the locus

    :param tx: transcript to add
    :type tx: Transcript
    """
    for y in [x.payload for x in self.g.get_nodes()]:
      if tx.id in [z.id for z in y]:
        sys.stderr.write("WARNING tx is already in graph\n")
        return True
    # transcript isn't part of graph yet
    n = seqtools.graph.Node([tx])

    other_nodes = self.g.get_nodes()
    self.g.add_node(n)
    # now we need to see if its connected anywhere
    for n2 in other_nodes:
     tx2s = n2.payload
     for tx2 in tx2s:
      # do exon overlap
      er = self.merge_rules.get_exon_rules()
      # if we are doing things by exon
      if (self.merge_rules.get_use_single_exons() and (tx.get_exon_count() == 1 or tx2.get_exon_count() == 1)) or \
         (self.merge_rules.get_use_multi_exons() and (tx.get_exon_count() > 1 and tx2.get_exon_count() > 1)):
        eo = tx.exon_overlap(tx2,multi_minover=er['multi_minover'],multi_endfrac=er['multi_endfrac'],multi_midfrac=er['multi_midfrac'],single_minover=er['single_minover'],single_frac=er['single_frac'])
        if self.merge_rules.get_merge_type() == 'is_compatible':
          if eo.is_compatible():
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)
        elif self.merge_rules.get_merge_type() == 'is_subset':
          r = eo.is_subset()
          if r == 2 or r == 1:
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
          if r == 3 or r == 1:
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)
        elif self.merge_rules.get_merge_type() == 'is_full_overlap':
          if eo.is_full_overlap():
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)
        elif self.merge_rules.get_merge_type() == 'is_any_overlap':
          if eo.match_exon_count() > 0:
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)        
            
      if self.merge_rules.get_use_junctions():
        # do junction overlap
        jo = tx.junction_overlap(tx2,self.merge_rules.get_juntol())
        #print jo.match_junction_count()
        if self.merge_rules.get_merge_type() == 'is_compatible':
          if jo.is_compatible():
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)
        elif self.merge_rules.get_merge_type() == 'is_subset':
          r = jo.is_subset()
          if r == 2 or r == 1:
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
          if r == 3 or r == 1:
            self.g.add_edge(Seqtools.graph.Edge(n2,n),verbose=False)
        elif self.merge_rules.get_merge_type() == 'is_full_overlap':
          if jo.is_full_overlap():
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)
        elif self.merge_rules.get_merge_type() == 'is_any_overlap':
          if jo.match_junction_count() > 0:
            self.g.add_edge(seqtools.graph.Edge(n,n2),verbose=False)
            self.g.add_edge(seqtools.graph.Edge(n2,n),verbose=False)        
    return True