def update_loci(self):
    """Goes through and combines loci until we have one set meeting our overlap definition"""
    # Create sub-loci for each chromosome
    lbc = {}
    chroms = sorted([x.range.chr for x in self.loci])
    for chrom in chroms: lbc[chrom] = Loci()
    for x in self.loci: lbc[x.range.chr].add_locus(x)
    for chrom in sorted(lbc.keys()):
      if self.verbose: 
        lbc[chrom].verbose = True
        sys.stderr.write(chrom+"\n")
      lbc[chrom].overhang = self.overhang
      lbc[chrom].use_direction = self.use_direction
      lbc[chrom].merge_down_loci()
    self.loci = []
    for chrom in sorted(lbc.keys()):
      for locus in lbc[chrom].loci:  self.loci.append(locus)