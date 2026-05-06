def subtract_range_array(bed1,beds2,is_sorted=False):
  """subtract several ranges from a range, returns array1 - (all of array2)

  :param bed1: 
  :param beds2: subtract all these beds from bed1
  :param is_sorted: has it been sorted already? Default (False)
  :type bed1: Bed or GenomicRange
  :type beds2: Bed[] or GenomicRange[]
  :param is_sorted: bool

  """
  if not is_sorted: beds2 = sort_ranges(beds2)
  output = [bed1.copy()]  
  mink = 0
  for j in range(0,len(beds2)):
    temp = []
    if mink > 0: temp = output[0:mink]
    for k in range(mink,len(output)):
      cmpv = output[k].cmp(beds2[j])
      if cmpv ==-1: mink=k
      temp += output[k].subtract(beds2[j])
    #for nval in [x.subtract(beds2[j]) for x in output]:
    #  temp += nval
    output = temp
  return output