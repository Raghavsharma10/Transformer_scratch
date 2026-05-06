def ranges_to_coverage(rngs,threads=1):
  """take a list of ranges as an input
  output a list of ranges and the coverage at each range
  :param rngs: bed ranges on a single chromosome. not certain about that single chromosome requirement
  :type rngs: GenomicRange[] or Bed[]
  :param threads: Not currently being used
  :type threads: int

  :return: out is the non-overlapping bed ranges with the edition of depth
  :rtype: GenomicRange[]
  """
  def do_chr(rngs):
    """do one chromosomes sorting
    :param rngs:
    :type rngs: GenomicRange[]
    """
    #starts = sorted(range(0,len(rngs)), key=lambda x: rngs[x].start)
    #print starts
    #ends = sorted(range(0,len(rngs)), key=lambda x: rngs[x].end)
    start_events = [x.start for x in rngs]
    end_events = [x.end+1 for x in rngs]
    indexed_events = {}
    for e in start_events:
      if e not in indexed_events: indexed_events[e] = {'starts':0,'ends':0}
      indexed_events[e]['starts']+=1
    for e in end_events:
      if e not in indexed_events: indexed_events[e] = {'starts':0,'ends':0}
      indexed_events[e]['ends']+=1
    cdepth = 0
    pstart = None
    pend = None
    outputs = []
    ordered_events = sorted(indexed_events.keys())
    for loc in ordered_events:
      prev_depth = cdepth # where we were
      # see where we are before the change
      cdepth += indexed_events[loc]['starts']
      cdepth -= indexed_events[loc]['ends']
      if prev_depth > 0 and prev_depth != cdepth:
        outputs.append([rngs[0].chr,pstart,loc-1,prev_depth]) # output what was before this if we are in something
      if prev_depth != cdepth or cdepth == 0:
        pstart = loc
    #print outputs
    return outputs
  
  class Queue:
    """Simple class to be able to use get function to retreive a value"""
    def __init__(self,val):
      self.val = [val]
    def get(self):
      return self.val.pop(0)

  ### START MAIN ####
  srngs = sort_genomic_ranges(rngs)
  # get the leftmost unique range
  chr = srngs[0].chr
  buffer = []
  results = []
  prelim = []
  for b in srngs:
    if b.chr != chr:
      rs = do_chr(buffer[:])
      for r in rs:  
        results.append(GenomicRange(r[0],r[1],r[2]))
        results[-1].set_payload(r[3])
      buffer = []
    buffer.append(b)
    chr = b.chr
  if len(buffer) > 0:
    rs = do_chr(buffer[:])
    for r in rs: 
      results.append(GenomicRange(r[0],r[1],r[2]))
      results[-1].set_payload(r[3])
  return results