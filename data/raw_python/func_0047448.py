def GenomicRangeFromString(range_string,payload=None,dir=None):
  """Constructor for a GenomicRange object that takes a string"""
  m = re.match('^(.+):(\d+)-(\d+)$',range_string)
  if not m:  
    sys.stderr.write("ERROR bad genomic range string\n"+range_string+"\n")
    sys.exit()
  chr = m.group(1)
  start = int(m.group(2))
  end = int(m.group(3))
  return GenomicRange(chr,start,end,payload,dir)