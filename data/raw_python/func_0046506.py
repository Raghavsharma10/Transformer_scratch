def trim_ordered_range_list(ranges,start,finish):
  """A function to help with slicing a mapping
     Start with a list of ranges and get another list of ranges constrained by start (0-indexed) and finish (1-indexed)

     :param ranges: ordered non-overlapping ranges on the same chromosome
     :param start: start 0-indexed
     :param finish: ending 1-indexed
     :type ranges: GenomicRange []
     :type start: Int
     :type finish: Int
     :return: non-overlapping ranges on same chromosome constrained by start and finish
     :rtype: GenomicRange []
  """
  z = 0
  keep_ranges = []
  for inrng in self.ranges:
    z+=1
    original_rng = inrng
    rng = inrng.copy() # we will be passing it along and possibly be cutting it
    done = False;
    #print 'exon length '+str(rng.length())
    if start >= index and start < index+original_rng.length(): # we are in this one
      rng.start = original_rng.start+(start-index) # fix the start
      #print 'fixstart '+str(original_rng.start)+' to '+str(rng.start)
    if finish > index and finish <= index+original_rng.length():
      rng.end = original_rng.start+(finish-index)-1
      done = True
      #print 'fixend '+str(original_rng.end)+' to '+str(rng.end)
 
    if finish <= index+original_rng.length(): # we are in the last exon we need
      index+= original_rng.length()
      keep_ranges.append(rng)
      break
    if index+original_rng.length() < start: # we don't need any bases from this
      index += original_rng.length()
      continue # we don't use this exon
    keep_ranges.append(rng)
    index += original_rng.length()
    if index > finish: break
    if done: break
  return keep_ranges