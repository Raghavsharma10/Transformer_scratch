def tama_read(self,filename):
    """
    Parse the science segments from a tama list of locked segments contained in
                file.
    @param filename: input text file containing a list of tama segments.
    """
    self.__filename = filename
    for line in open(filename):
      columns = line.split()
      id = int(columns[0])
      start = int(math.ceil(float(columns[3])))
      end = int(math.floor(float(columns[4])))
      dur = end - start

      x = ScienceSegment(tuple([id, start, end, dur]))
      self.__sci_segs.append(x)