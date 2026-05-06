def get_target_transcript(self,min_intron=1):
    """Get the mapping of to the target strand

    :returns: Transcript mapped to target
    :rtype: Transcript

    """
    if min_intron < 1: 
      sys.stderr.write("ERROR minimum intron should be 1 base or longer\n")
      sys.exit()
    #tx = Transcript()
    rngs = [self.alignment_ranges[0][0].copy()]
    #rngs[0].set_direction(None)
    for i in range(len(self.alignment_ranges)-1):
      dist = self.alignment_ranges[i+1][0].start - rngs[-1].end-1
      #print 'dist '+str(dist)
      if dist >= min_intron:
        rngs.append(self.alignment_ranges[i+1][0].copy())
        #rngs[-1].set_direction(None)
      else:
        rngs[-1].end = self.alignment_ranges[i+1][0].end
    tx = Transcript(rngs,options=Transcript.Options(
         direction=self.strand,
         name = self.alignment_ranges[0][1].chr,
         gene_name = self.alignment_ranges[0][1].chr
                                                  ))
    #tx.set_exons_and_junctions_from_ranges(rngs)
    #tx.set_range()
    #tx.set_strand(self.get_strand())
    #tx.set_transcript_name(self.get_alignment_ranges()[0][1].chr)
    #tx.set_gene_name(self.get_alignment_ranges()[0][1].chr)
    return tx