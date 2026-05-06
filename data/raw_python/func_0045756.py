def merge_ranges(inranges,already_sorted=False):
  """from a list of genomic range or bed entries, whether or not they are already sorted, 
     make a flattend range list of ranges where if they overlapped, they are now joined
     (not yet) The new range payloads will be the previous ranges

  :param inranges:
  :param already_sorted: has this already been sorted (defaults to False)
  :type inranges: GenomicRange[]
  :type already_sorted: bool

  :return: sorted ranges
  :rtype: GenomicRange[]

  """
  if not already_sorted: inranges = sort_ranges(inranges)
  prev = None
  outputs = []
  merged = False
  for rng in inranges:
    #nrng = rng.copy()
    #nrng.set_payload([])
    #nrng.get_payload().append(rng)
    merged = False
    if len(outputs) > 0:
      if rng.overlaps(outputs[-1]) or rng.adjacent(outputs[-1]):
        nrng = rng.merge(outputs[-1])
        #nrng.set_payload(prev.get_payload())
        #nrng.get_payload().append(rng)
        outputs[-1] = nrng
        merged = True
    if not merged:
      outputs.append(rng.copy())
    #prev = nrng
  #if not merged: outputs.append(prev)
  return sort_ranges(outputs)