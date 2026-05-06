def main(args):

  """ First read the genome """
  sys.stderr.write("reading reference genome\n")
  ref = FASTAData(open(args.genome).read())
  sys.stderr.write("read in "+str(len(ref.keys()))+" chromosomes\n")

  """ Next make the transcriptome """
  txome = {}
  sys.stderr.write("write the transcriptome\n")
  inf = None
  if args.gpd[-3:] == '.gz':
    inf = gzip.open(args.gpd)
  else:
    inf = open(args.gpd)
  stream = GPDStream(inf)
  tof = open(args.tempdir+'/transcriptome.fa','w')
  z = 0
  for gpd in stream:
    z += 1
    if gpd.transcript_name in txome:
      sys.stderr.write("WARNING already have a transcript "+gpd.transcript_name+" ignoring line "+str(z)+" of the gpd\n")
      continue
    txome[gpd.transcript_name] = gpd.gene_name
    tof.write('>'+gpd.transcript_name+"\n"+str(gpd.get_sequence(ref))+"\n")
  tof.close()
  inf.close()
  sys.stderr.write("wrote "+str(len(txome.keys()))+" transcripts\n")

  """Build the salmon index"""
  sys.stderr.write("building a salmon index\n")
  cmd = 'salmon index -p '+str(args.numThreads)+' -t '+args.tempdir+'/transcriptome.fa -i '+args.tempdir+'/salmon_index'
  p = Popen(cmd.split())
  p.communicate()
  sys.stderr.write("finished building the index\n")

  """Use the index to quanitfy"""
  sys.stderr.write("quanitfy reads\n")
  reads = ''
  if args.rU:
    reads = '-r '+args.rU
  else:
    reads = '-1 '+args.r1+' -2 '+args.r2
  cmd = 'salmon quant -p '+str(args.numThreads)+' -i '+args.tempdir+'/salmon_index -l A '+reads+' -o '+args.tempdir+'/output_quant'
  p = Popen(cmd.split())
  p.communicate()
  sys.stderr.write("finished quanitfying\n")

  """Now parse the salmon output to add gene name"""
  salmon = {}
  with open(args.tempdir+'/output_quant/quant.sf') as inf:
    header = inf.readline()
    for line in inf:
      f = line.rstrip().split("\t")
      # by each transcript name hold a data strcture of the information
      salmon[f[0]] = {'name':f[0],'length':int(f[1]),'EffectiveLength':float(f[2]),'TPM':float(f[3]),'NumReads':float(f[4])}
  genes = {}
  for name in salmon:
    gene = txome[name]
    if gene not in genes: genes[gene] = []
    genes[gene].append(salmon[name])
  genetot = {}
  for gene in genes:
    tot = sum([x['TPM'] for x in genes[gene]])
    genetot[gene] = tot
  ordered_gene_names = sorted(genetot.keys(), key=lambda x: genetot[x],reverse=True)
      
  """Collected enough information to make output"""
  sys.stderr.write("generating output\n")
  of = sys.stdout
  if args.output != '-':
    of = open(args.output,'w')
  of.write("geneName\ttranscriptName\tlength\tEffectiveLength\ttxTPM\tNumReads\tgeneTPM\n")
  for gene in ordered_gene_names:
    txs = sorted(genes[gene],key=lambda x: x['TPM'],reverse=True)
    for tx in txs:
      of.write(gene+"\t"+tx['name']+"\t"+str(tx['length'])+"\t"+str(tx['EffectiveLength'])+"\t"+str(tx['TPM'])+"\t"+str(tx['NumReads'])+"\t"+str(genetot[gene])+"\n")
  of.close()

  # Temporary working directory step 3 of 3 - Cleanup
  if not args.specific_tempdir:
    rmtree(args.tempdir)