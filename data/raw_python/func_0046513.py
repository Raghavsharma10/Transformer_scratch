def get_transcript(self,exon_bounds='max'):
    """Return a representative transcript object"""
    out = Transcript()
    out.junctions = [x.get_junction() for x in self.junction_groups]
    # check for single exon transcript
    if len(out.junctions) == 0:
      leftcoord = min([x.exons[0].range.start for x in self.transcripts])
      rightcoord = max([x.exons[-1].range.end for x in self.transcripts])
      e = Exon(GenomicRange(x.exons[0].range.chr,leftcoord,rightcoord))
      e.set_is_leftmost()
      e.set_is_rightmost()
      out.exons.append(e)
      return out
    # get internal exons
    self.exons = []
    for i in range(0,len(self.junction_groups)-1):
      j1 = self.junction_groups[i].get_junction()
      j2 = self.junction_groups[i+1].get_junction()
      e = Exon(GenomicRange(j1.right.chr,j1.right.end,j2.left.start))
      e.set_left_junc(j1)
      e.set_right_junc(j2)
      #print str(i)+" to "+str(i+1)
      out.exons.append(e)
    # get left exon
    left_exons = [y for y in [self.transcripts[e[0]].junctions[e[1]].get_left_exon() for e in self.junction_groups[0].evidence] if y]
    if len(left_exons) == 0:
      sys.stderr.write("ERROR no left exon\n")
      sys.exit()
    e_left = Exon(GenomicRange(out.junctions[0].left.chr,\
                               min([x.range.start for x in left_exons]),
                               out.junctions[0].left.start))
    e_left.set_right_junc(out.junctions[0])
    out.exons.insert(0,e_left)
    # get right exon
    right_exons = [y for y in [self.transcripts[e[0]].junctions[e[1]].get_right_exon() for e in self.junction_groups[-1].evidence] if y]
    if len(right_exons) == 0:
      sys.stderr.write("ERROR no right exon\n")
      sys.exit()
    e_right = Exon(GenomicRange(out.junctions[-1].right.chr,\
                               out.junctions[-1].right.end,\
                               max([x.range.end for x in right_exons])))
    e_right.set_left_junc(out.junctions[-1])
    out.exons.append(e_right)
    return out