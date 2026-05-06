def annotated_reads(txome,sorted_read_stream,output,args):
   if not os.path.exists(args.tempdir+'/'+output):
      os.makedirs(args.tempdir+'/'+output)
   ts = OrderedStream(iter(txome.transcript_stream()))
   rs = OrderedStream(GPDStream(sorted_read_stream))
   mls = MultiLocusStream([ts,rs])
   
   aof = gzip.open(args.tempdir+'/'+output+'/annotated.txt.gz','w')
   of1 = gzip.open(args.tempdir+'/'+output+'/unannotated_multiexon.gpd.gz','w')
   of2 = gzip.open(args.tempdir+'/'+output+'/partial_annotated_multiexon.gpd.gz','w')
   of3 = gzip.open(args.tempdir+'/'+output+'/unannotated_singleexon.gpd.gz','w')
   z = 0
   for ml in mls:
      refs, reads = ml.payload
      if len(refs) == 0: continue
      """Check and see if we have single exons annotated"""
      single_exon_refs = [x for x in refs if x.get_exon_count()==1]
      single_exon_reads = [x for x in reads if x.get_exon_count()==1]
      #print str(len(single_exon_refs))+" "+str(len(single_exon_reads))
      unannotated_single_exon_reads = []
      for seread in single_exon_reads:
         ovs  = [(x.exons[0].length,seread.exons[0].length,x.exons[0].overlap_size(seread.exons[0]),x) for x in single_exon_refs if x.exons[0].overlaps(seread.exons[0])]
         #check for minimum overlap
         ovs = [x for x in ovs if x[2] >= args.single_exon_minimum_overlap]
         ovs = [x for x in ovs if float(x[2])/float(max(x[0],x[1])) >= args.single_exon_mutual_overlap]
         ovs = sorted(ovs,key=lambda x: float(x[2])/float(max(x[0],x[1])),reverse=True)
         if len(ovs) == 0: 
            unannotated_single_exon_reads.append(seread)
            continue
         best_ref = ovs[0][3]
         aof.write(seread.name+"\t"+best_ref.name+"\tSE"+"\n")

      """May want an optional check for any better matches among exons"""
      #now we can look for best matches among multi-exon transcripts
      reads = [x for x in reads if x.get_exon_count() > 1]
      multiexon_refs = [x for x in refs if x.get_exon_count() > 1]
      unannotated_multi_exon_reads = []
      partial_annotated_multi_exon_reads = []
      for read in reads:
         # we dont' need to have multiple exons matched. one is enough to call
         ovs = [y for y in [(x,x.exon_overlap(read,
                                              multi_minover=args.multi_exon_minimum_overlap,
                                              multi_endfrac=args.multi_exon_end_frac,
                                              multi_midfrac=args.multi_exon_mid_frac,
                                              multi_consec=False)) for x in multiexon_refs] if y[1]]
         for o in ovs: o[1].analyze_overs()
         full =  sorted([x for x in ovs if x[1].is_subset()==1],
                        key = lambda y: float(y[1].overlap_size())/float(max(y[1].tx_obj1.length,y[1].tx_obj2.length)),
                        reverse=True
                       )
         if len(full) > 0:
            aof.write(read.name+"\t"+full[0][0].name+"\tfull"+"\n")
            continue
         subset =  sorted([x for x in ovs if x[1].is_subset()==2],
                          key = lambda y: (y[1].match_exon_count(),
                                           y[1].min_overlap_fraction()),
                          reverse = True
                         )
         if len(subset) > 0:
            aof.write(read.name+"\t"+subset[0][0].name+"\tpartial"+"\n")
            continue
         #check for supersets
         superset = sorted([x for x in ovs if x[1].is_subset()==3],
                          key = lambda y: (y[1].match_exon_count(),
                                           y[1].min_overlap_fraction()),
                          reverse = True
                         )
         if len(superset) > 0:
            partial_annotated_multi_exon_reads.append((read,superset[0][0]))
            #print read.name+"\t"+superset[0][0].name+"\tsuper"
            continue
         #check for noncompatible overlaps
         overset = sorted([x for x in ovs if x[1].match_exon_count > 0],
                          key = lambda y: (y[1].consecutive_exon_count(),
                                           y[1].min_overlap_fraction()),
                          reverse = True
                         )
         #print [(x[1].consecutive_exon_count(), x[1].min_overlap_fraction()) for x in overset]
         if len(overset) > 0:
            partial_annotated_multi_exon_reads.append((read,overset[0][0]))
            #print read.name+"\t"+overset[0][0].name+"\tover"
            continue
         unannotated_multi_exon_reads.append(read)
      """Now we have partially annotated multi and unannotated multi and unannotated single"""
      if len(unannotated_multi_exon_reads) > 0:
         sys.stderr.write(str(z)+" "+str(len(unannotated_multi_exon_reads))+"   \r")
         d = Deconvolution(downsample(unannotated_multi_exon_reads,args.downsample_locus))
         groups = d.parse(tolerance=20,downsample=args.downsample)
         for tx in groups:
            z+=1
            of1.write(tx.get_gpd_line()+"\n")
      if len(partial_annotated_multi_exon_reads) > 0:
         sys.stderr.write(str(z)+" "+str(len(partial_annotated_multi_exon_reads))+"   \r")
         ### set the direction of the transcript
         for v in partial_annotated_multi_exon_reads:
            v[0].set_strand(v[1].direction)
         ### set the gene name of the transcript
         for v in partial_annotated_multi_exon_reads:
            v[0].set_gene_name(v[1].gene_name)         
         d = Deconvolution(downsample([x[0] for x in partial_annotated_multi_exon_reads],args.downsample_locus))
         groups = d.parse(tolerance=20,downsample=args.downsample,use_gene_names=True)
         for tx in groups:
            z += 1
            of2.write(tx.get_gpd_line()+"\n")
      """Do the unannotated single exon reads"""
      g = Graph()
      for r in unannotated_single_exon_reads:
         if len([x for x in partial_annotated_multi_exon_reads if x[0].overlaps(r)]) > 0: continue
         if len([x for x in unannotated_multi_exon_reads if x.overlaps(r)]) > 0: continue
         if len([x for x in txome.transcripts if x.overlaps(r)]) > 0: continue
         g.add_node(Node(r))
      for i in range(0,len(g.nodes)):
         for j in range(0,len(g.nodes)):
            if i == j: continue
            if g.nodes[i].payload.overlaps(g.nodes[j].payload):
               g.add_edge(Edge(g.nodes[i],g.nodes[j]))
      g.merge_cycles()
      for r in g.roots:
         se = []
         se += r.payload_list
         children = g.get_children(r)
         for child in children:  se += child.payload_list
         rng = GenomicRange(se[0].exons[0].chr,
                            min([x.exons[0].start for x in se]),
                            max([x.exons[0].end for x in se]))
         tx = Transcript([rng],Transcript.Options(direction='+'))
         of3.write(tx.get_gpd_line()+"\n")

   of1.close()
   of2.close()
   of3.close()
   sys.stderr.write("\n")      
   """Sort our progress"""
   sys.stderr.write("sort transcriptome\n")
   sort_gpd(args.tempdir+'/'+output+'/partial_annotated_multiexon.gpd.gz',args.tempdir+'/'+output+'/partial_annotated_multiexon.sorted.gpd.gz',args)
   sort_gpd(args.tempdir+'/'+output+'/unannotated_multiexon.gpd.gz',args.tempdir+'/'+output+'/unannotated_multiexon.sorted.gpd.gz',args)
   sort_gpd(args.tempdir+'/'+output+'/unannotated_singleexon.gpd.gz',args.tempdir+'/'+output+'/unannotated_singleexon.sorted.gpd.gz',args)
   """We still have the unannotated single exon reads to deal with"""