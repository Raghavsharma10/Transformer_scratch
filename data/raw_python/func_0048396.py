def write_index(path,index_file,verbose=False,samtools=False):
  """ Index file is a gzipped TSV file with these fields:

  1. qname
  2. target range
  3. bgzf file block start
  4. bgzf inner block start
  5. aligned base count
  6. flag

  :param path: bamfile
  :param index_file: bam index to write
  :param verbose: default False
  :param samtools: use samtools default False
  :type path:
  :type index_file:
  :type verbose: bool
  :type samtools: bool
  """
  if verbose:
    sys.stderr.write("scanning for primaries\n")
  reads = {}
  z = 0
  # force use of primary alignment flag if its not already used
  # require one and only one primary alignment for each read (or mate)
  fail_primary = False
  b2 = None
  if samtools:
    b2 = SamtoolsBAMStream(path)
  else:
    b2 = BAMFile(path)

  for e in b2:
    z+=1
    if verbose:
      if z %1000==0: sys.stderr.write(str(z)+"\r")
    name = e.value('qname')
    if name not in reads:
      reads[name] = {}
    type = 'u'
    if e.check_flag(64):
      type = 'l' #left mate
    elif e.check_flag(128):
      type = 'r' #right mate
    if not e.check_flag(2304):
      if type not in reads[name]: reads[name][type] = 0
      reads[name][type] += 1 # we have one
      if reads[name][type] > 1: 
        fail_primary = True
        break # too many primaries set to be useful
  # see if we have one primary set for each read
  for name in reads:
    for type in reads[name]:
      if reads[name][type] != 1:
        fail_primary = True
  if verbose:
    sys.stderr.write("\n")
  if fail_primary:
    sys.stderr.write("Failed to find a single primary for each read (or each mate).  Reading through bam to find best.\n")
    best = {}
    # must find the primary for each

    b2 = None
    if samtools:
      b2 = SamtoolsBAMStream(path)
    else:
      b2 = BAMFile(path)
    z = 0
    for e in b2:
      z += 1
      if verbose:
        if z %1000==0: sys.stderr.write(str(z)+"\r")
      name = e.value('qname')
      type = 'u'
      if e.check_flag(64):
        type = 'l' #left mate
      elif e.check_flag(128):
        type = 'r' #right mate
      if name not in best: best[name] = {}
      # get length
      l = 0
      if e.is_aligned():
        l = e.get_aligned_bases_count()
      if type not in best[name]: best[name][type] = {'line':z,'bpcnt':l}
      if l > best[name][type]['bpcnt']: 
        best[name][type]['bpcnt'] = l
        best[name][type]['line'] = z
    bestlinenumbers = set()
    for name in best:
      for type in best[name]:
        bestlinenumbers.add(best[name][type]['line'])
    if verbose:
      sys.stderr.write("\n")
  of = None
  try:
    of = gzip.open(index_file,'w')
  except IOError:
    sys.sterr.write("ERROR: could not find or create index\n")
    sys.exit()


  b2 = None
  if samtools:
    b2 = SamtoolsBAMStream(path)
  else:
    b2 = BAMFile(path)
  z = 0
  for e in b2:
    z+=1
    if verbose:
      if z%1000==0:
        sys.stderr.write(str(z)+" reads indexed\r")
    myflag = e.value('flag')
    if fail_primary: # see if this should be a primary
      if z not in bestlinenumbers:
        myflag = myflag | 2304
    rng = e.get_target_range()
    if rng: 
      l = e.get_aligned_bases_count()
      of.write(e.value('qname')+"\t"+rng.get_range_string()+"\t"+str(e.get_block_start())+"\t"+str(e.get_inner_start())+"\t"+str(l)+"\t"+str(myflag)+"\n")
    else: of.write(e.value('qname')+"\t"+''+"\t"+str(e.get_block_start())+"\t"+str(e.get_inner_start())+"\t"+'0'+"\t"+str(myflag)+"\n")
  sys.stderr.write("\n")
  of.close()