def read_entry(self):
    """read the next bed entry from the stream"""
    line = self.fh.readline()
    if not line: return None
    m = re.match('([^\t]+)\t(\d+)\t(\d+)\t*(.*)',line.rstrip())
    if not m:
      sys.stderr.write("ERROR: unknown line in bed format file\n"+line+"\n")
      sys.exit()
    g = GenomicRange(m.group(1),int(m.group(2))+1,int(m.group(3)))
    if len(m.group(4)) > 0:
      g.set_payload(m.group(4))
    return g