def parse(self,tolerance=0,downsample=None,evidence=2,use_gene_names=False):
      """Divide out the transcripts.  allow junction tolerance if wanted"""
      g = Graph()
      nodes = [Node(x) for x in self._transcripts]
      for n in nodes: g.add_node(n)
      for i in range(0,len(nodes)):
         for j in range(0,len(nodes)):
            if i == j: continue
            jov = nodes[i].payload.junction_overlap(nodes[j].payload,tolerance)
            sub = jov.is_subset()
            if not sub: continue
            if sub == 1:
               g.add_edge(Edge(nodes[i],nodes[j]))
               g.add_edge(Edge(nodes[j],nodes[i]))
            if sub == 2:
               g.add_edge(Edge(nodes[i],nodes[j]))
      g.merge_cycles()
      roots = g.roots
      groups = []
      for r in roots:
         g2 = g.get_root_graph(r)
         c = CompatibleGraph(g2,tolerance,downsample,evidence,use_gene_names=use_gene_names)
         groups.append(c)
      return groups