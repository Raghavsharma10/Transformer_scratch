def main(args):
  """first we need sorted genepreds"""
  cmd = ['sort',args.reference,'--gpd','--tempdir',args.tempdir,'--threads',
         str(args.threads),'-o',args.tempdir+'/ref.sorted.gpd']
  sys.stderr.write(cmd+"\n")
  gpd_sort(cmd)
  cmd = ['sort',args.gpd,'--gpd','--tempdir',args.tempdir,'--threads',
         str(args.threads),'-o',args.tempdir+'/my.sorted.gpd']
  sys.stderr.write(cmd+"\n")
  gpd_sort(cmd)
  rstream = GPDStream(open(args.tempdir+'/ref.sorted.gpd'))
  mstream = GPDStream(open(args.tempdir+'/my.sorted.gpd'))
  stream = MultiLocusStream([rstream,mstream])
  of = sys.stdout
  if args.output != '-':
    if args.output[-3:] == '.gz': of = gzip.open(args.output,'w')
    else: of = open(args.output,'w')
  for locus_rng in stream:
    (rgpds, mgpds) = locus_rng.get_payload()
    if len(mgpds) == 0: continue
    sys.stderr.write(locus_rng.get_range_string()+" "+str(len(rgpds))+" "+str(len(mgpds))+"     \r")
    ref_juncs = {}
    for ref in rgpds: ref_juncs[ref.get_junction_string()] = ref
    annotated = []
    unannotated = []
    annotated = [ref_juncs[x.get_junction_string()] for x in mgpds if x.get_exon_count() > 1 and x.get_junction_string() in ref_juncs]
    unannotated = [x for x in mgpds if x.get_exon_count() > 1 and x.get_junction_string() not in ref_juncs]
    # now unannotated needs an annotation.
    my_unannotated = [x for x in mgpds if x.get_exon_count() == 1]
    single_reference = [x for x in rgpds if x.get_exon_count() == 1]
    single_annotated = []
    single_unannotated = []
    #print len(single_reference)
    #print len(single_unannotated)
    for gpd in my_unannotated:
      overs = sorted([x for x in single_reference if x.overlap_size(gpd) > 0],\
      key=lambda y: y.avg_mutual_coverage(gpd), reverse=True)
      if len(overs) > 0:
        single_annotated.append(overs[0])
      else: single_unannotated.append(gpd)
    # now annotated and single_annotated are done
    unannotated += single_unannotated
    # now single or multi we need to annotated unanotated
    gene_annotated = []
    no_annotation = []
    for m in unannotated:
      overs = sorted([x for x in rgpds if x.overlap_size(m) > 0],\
      key=lambda y: y.avg_mutual_coverage(m), reverse=True)
      if len(overs) > 0:
        gname = overs[0].value('gene_name')
        f = overs[0].get_gpd_line().rstrip().split("\t")
        f[0] = gname
        f[1] = str(uuid.uuid4())
        g = GPD("\t".join(f))
        gene_annotated.append(g)
      else: no_annotation.append(m)
    finished = []
    # now we need to annotate no_annotation
    while len(no_annotation) > 0:
      m = no_annotation.pop(0)
      matched = False
      for i in range(0,len(finished)):
        if len([x for x in finished[i] if x.overlap_size(m) > 0]) > 0:
          finished[i].append(m)
          matched = True
          break
      if not matched:  finished.append([m])
    # now finished has gene groups
    original = []
    for group in finished:
      gname = str(uuid.uuid4())
      for member in group:
        tname = str(uuid.uuid4())
        f = member.get_gpd_line().rstrip().split("\t")
        f[0] = gname
        f[1] = tname
        g = GPD("\t".join(f))
        original.append(g)
    for gpd in original + annotated + single_annotated + gene_annotated:
      of.write(gpd.get_gpd_line()+"\n")
  of.close()
  sys.stderr.write("\n")
  # Temporary working directory step 3 of 3 - Cleanup
  if not args.specific_tempdir:
    rmtree(args.tempdir)