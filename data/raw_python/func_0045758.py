def subtract_ranges(r1s,r2s,already_sorted=False):
  """Subtract multiple ranges from a list of ranges

  :param r1s: range list 1
  :param r2s: range list 2
  :param already_sorted: default (False)
  :type r1s: GenomicRange[]
  :type r2s: GenomicRange[]

  :return: new range r1s minus r2s
  :rtype: GenomicRange[]
  """
  from seqtools.stream import MultiLocusStream
  if not already_sorted:
    r1s = merge_ranges(r1s)
    r2s = merge_ranges(r2s)
  outputs = []
  mls = MultiLocusStream([BedArrayStream(r1s),BedArrayStream(r2s)])
  tot1 = 0
  tot2 = 0
  for loc in mls:
    #[beds1,beds2] = loc.get_payload()
    v = loc.payload
    #print v
    [beds1,beds2] =v
    beds1 = beds1[:]
    beds2 = beds2[:]
    if len(beds1)==0:
      continue
    if len(beds2)==0:
      outputs += beds1
      continue
    #this loop could be made much more efficient
    mapping = {} #keyed by beds1 index stores list of overlaping beds2 indecies
    for i in range(0,len(beds1)):
      mapping[i] = []
    beds2min = 0
    beds2max = len(beds2)
    for i in range(0,len(beds1)):
      for j in range(beds2min,beds2max):
        cmpval = beds1[i].cmp(beds2[j])
        if cmpval == -1:
          beds2min = j+1
        elif cmpval == 0:
          mapping[i].append(j)
        else:
          break
    for i in range(0,len(beds1)):
      if len(mapping[i])==0: outputs += beds1
      else:
        outputs += subtract_range_array(beds1[i],[beds2[j] for j in mapping[i]],is_sorted=True)
    #while len(beds2) > 0:
    #  b2 = beds2.pop(0)
    #  vs = [x.subtract(b2) for x in beds1]
    #  tot = []
    #  for res in vs:
    #    tot = tot + res
    #  beds1 = tot
    #print "subtract "+str(len(beds1))+"\t"+str(len(beds2))
    #print beds1[0].get_range_string()
  #outputs = merge_ranges(outputs)
  #print [x.get_range_string() for x in outputs]

  return merge_ranges(outputs)