def main(args):
   """Read any reference transcriptome we can"""
   txome = Transcriptome()
   #read the reference gpd if one is gven
   if args.reference: 
      rinf = None
      if re.search('\.gz$',args.reference):
         rinf = gzip.open(args.reference)
      else:
         rinf = open(args.reference)
      sys.stderr.write("Reading in reference\n")
      z = 0
      # populate txome with reference transcripts for each chromosome
      for line in rinf:
         z += 1
         gpd = GPD(line)
         gpd.set_payload(z)
         if z%100 == 0:  sys.stderr.write(str(z)+"          \r")
         txome.add_transcript(gpd)
      rinf.close()
      sys.stderr.write(str(z)+"          \r")
      sys.stderr.write("\n")
   txome.sort_transcripts()
   sys.stderr.write("Buffering mappings\n")
   inf = sys.stdin
   if args.input != '-':
      if re.search('\.gz$',args.input):
         inf = gzip.open(args.input)
      else:
         inf = open(args.input)
   tof = gzip.open(args.tempdir+'/reads.gpd.gz','w')
   for line in inf: tof.write(line.rstrip()+"\n")
   tof.close()
  
   sys.stderr.write("1. Process by overlapping locus.\n")
   annotated_reads(txome,gzip.open(args.tempdir+'/reads.gpd.gz'),'initial',args)

   #of.close()

   """Now sequences we should be able to annotate as partial have
      consensus transcripts.  Lets put those to annotation scrutiny again.
   """
   sys.stderr.write("2. Reannotated partially annotated by overlapping locus.\n")
   annotated_reads(txome,gzip.open(args.tempdir+'/initial/partial_annotated_multiexon.sorted.gpd.gz'),'repartial',args)


   txs = {}
   for transcript in txome.transcripts:
      txs[transcript.name] = transcript
   detected = {}
   tinf = gzip.open(args.tempdir+'/initial/annotated.txt.gz')
   for line in tinf:
      f = line.rstrip().split("\t")
      detected[f[1]] = True
   tinf.close()
   tinf = gzip.open(args.tempdir+'/repartial/annotated.txt.gz')
   remove_partial = {}
   for line in tinf:
      f = line.rstrip().split("\t")
      detected[f[1]] = True
      remove_partial[f[1]] = True
   tinf.close()
   tof = gzip.open(args.tempdir+'/candidate.gpd.gz','w')
   for name in detected:
      tof.write(txs[name].get_gpd_line()+"\n")
   tinf = gzip.open(args.tempdir+'/initial/partial_annotated_multiexon.sorted.gpd.gz')
   for line in tinf:
      f = line.rstrip().split("\t")
      if f[1] not in remove_partial:
         tof.write(line)
   tinf = gzip.open(args.tempdir+'/initial/unannotated_singleexon.sorted.gpd.gz')
   for line in tinf:
      f = line.rstrip().split("\t")
      f[0] = str(uuid.uuid4())
      tof.write("\t".join(f)+"\n")
   tinf.close()
   tof.close()
   sort_gpd(args.tempdir+'/candidate.gpd.gz',args.tempdir+'/candidate.sorted.gpd.gz',args)
   """We can ignore the partial annotations that have been detected"""
   ntxome = Transcriptome()
   tinf = gzip.open(args.tempdir+'/candidate.sorted.gpd.gz')
   for line in tinf: ntxome.add_transcript(GPD(line))     
   annotated_reads(ntxome,gzip.open(args.tempdir+'/initial/unannotated_multiexon.sorted.gpd.gz'),'unannotated',args)
   """now we know which unannotated reads actually have annotations"""
   tinf.close()

   tof = gzip.open(args.tempdir+'/final.gpd.gz','w')
   tinf = gzip.open(args.tempdir+'/candidate.sorted.gpd.gz')   
   for line in tinf: tof.write(line)
   tinf.close()
   tinf = gzip.open(args.tempdir+'/unannotated/unannotated_multiexon.sorted.gpd.gz')   
   for line in tinf: tof.write(line)
   tinf.close()
   tinf = gzip.open(args.tempdir+'/unannotated/partial_annotated_multiexon.sorted.gpd.gz')   
   for line in tinf: tof.write(line)
   tinf.close()
   tof.close()
   sort_gpd(args.tempdir+'/final.gpd.gz',args.tempdir+'/final.sorted.gpd.gz',args)
   """Prepare outputs"""
   of = sys.stdout
   if args.output:
      if re.search('\.gz$',args.output):
         of = gzip.open(args.output,'w')
      else:
         of = open(args.output,'w')
   tinf = gzip.open(args.tempdir+'/final.sorted.gpd.gz')
   for line in tinf: of.write(line)
   tinf.close()
   of.close()