def add_locus(self,inlocus):
    """ Adds a locus to our loci, but does not go through an update our locus sets yet"""
    if self.use_direction == True and inlocus.use_direction == False:
      sys.stderr.write("ERROR if using the direction in Loci, then every locus added needs use_direction to be True\n")
      sys.exit()
    self.loci.append(inlocus)
    return