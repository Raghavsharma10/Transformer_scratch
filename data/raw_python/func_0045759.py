def intersect_range_array(bed1,beds2,payload=None,is_sorted=False):
  """ Does not do a merge if the payload has been set

  :param bed1:
  :param bed2:
  :param payload: payload=1 return the payload of bed1 on each of the intersect set, payload=2 return the payload of bed2 on each of the union set, payload=3 return the payload of bed1 and bed2 on each of the union set
  :param is_sorted:
  :type bed1: GenomicRange
  :type bed2: GenomicRange
  :type payload: int
  :type is_sorted: bool
  """
  if not is_sorted: beds2 = sort_ranges(beds2)
  output = []
  for bed2 in beds2:
    cval = bed2.cmp(bed1)
    #print str(cval)+" "+bed1.get_range_string()+" "+bed2.get_range_string()
    if cval == -1: continue
    elif cval == 0:
      output.append(bed1.intersect(bed2))
      if payload==1:
        output[-1].set_payload(bed1.payload)
      if payload==2:
        output[-1].set_payload(bed2.payload)
    elif cval == 1: break
  if payload: return sort_ranges(output)
  return merge_ranges(output)