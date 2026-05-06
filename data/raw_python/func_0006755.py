def Blaster(inputfile, databases, db_path, out_path='.', min_cov=0.6,
            threshold=0.9, blast='blastn', cut_off=True):
   ''' BLAST wrapper method, that takes a simple input and produces a overview
   list of the hits to templates, and their alignments
   
   Usage
      >>> import os, subprocess, collections
      >>> from Bio.Blast import NCBIXML
      >>> from Bio import SeqIO
      >>> from string import maketrans
      >>> inputfile = 'test.fsa'
      >>> databases = ['enterobacteriaceae']
      >>> db_path   = '/path/to/databases/plasmidfinder/'
      >>> Blaster(inputfile, databases, db_path)
   '''
   min_cov = 100 * float(min_cov)
   threshold = 100 * float(threshold)
   
   # For alignment
   gene_align_query = dict() #will contain the sequence alignment lines
   gene_align_homo = dict()  #will contain the sequence alignment homolog string
   gene_align_sbjct = dict() #will contain the sequence alignment allele string
   results = dict()          #will contain the results
   
   for db in databases:
      # Adding the path to the database and output
      db_file = "%s/%s.fsa"%(db_path, db)
      os.system("mkdir -p %s/tmp"%(out_path))
      os.system("chmod 775 %s/tmp"%(out_path))
      out_file = "%s/tmp/out_%s.xml"%(out_path, db)
      
      # Running blast
      cmd = "%s -subject %s -query %s -out %s -outfmt '5' -perc_identity %s -dust 'no'"%(blast, db_file, inputfile, out_file, threshold)
      process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      out, err = process.communicate()
      
      # Getting the results
      result_handle = open(out_file)
      blast_records = NCBIXML.parse(result_handle)
      
      # Declaring variables for saving the results
      gene_results = dict() #will contain the results for each gene
      
      # For finding the best hits
      best_hsp = dict()
      
      # Keeping track of gene split
      gene_split = collections.defaultdict(dict)
      
      # Making the dicts for sequence outputs
      gene_align_query[db] = dict()
      gene_align_homo[db] = dict()
      gene_align_sbjct[db] = dict()
      
      # Parsing over the hits and only keeping the best
      for blast_record in blast_records:
         query = blast_record.query
         blast_record.alignments.sort(key = lambda align: -max((len(hsp.query) * (int(hsp.identities)/float(len(hsp.query))) for hsp in align.hsps)))
         for alignment in blast_record.alignments:
            # Setting the e-value as 1 and bit as 0 to get the best HSP fragment
            best_e_value = 1
            best_bit = 0
            for hsp in alignment.hsps:
               if hsp.expect < best_e_value or hsp.bits > best_bit:
                  best_e_value = hsp.expect
                  best_bit = hsp.bits
                  tmp = alignment.title.split(" ")
                  sbjct_header = tmp[1]
                  bit = hsp.bits
                  sbjct_length = alignment.length
                  sbjct_start = hsp.sbjct_start
                  sbjct_end = hsp.sbjct_end
                  gaps = hsp.gaps
                  query_string = str(hsp.query)
                  homo_string = str(hsp.match)
                  sbjct_string = str(hsp.sbjct)
                  contig_name = query.replace(">","")
                  query_start = hsp.query_start
                  query_end = hsp.query_end
                  HSP_length = len(query_string)
                  perc_ident = int(hsp.identities)/float(HSP_length) * 100
                  strand = 0
                  coverage = ((int(HSP_length) - int(gaps))/float(sbjct_length))
                  perc_coverage = ((int(HSP_length) - int(gaps))/float(sbjct_length)) * 100
                  if int(HSP_length) == int(sbjct_length):
                     cal_score = perc_ident * coverage * 100
                  else:
                     cal_score = perc_ident * coverage
                  hit_id = "%s:%s..%s:%s:%f"%(contig_name, query_start, query_end, sbjct_header, cal_score)
                  
                  # If the hit is on the other strand
                  if sbjct_start > sbjct_end:
                     tmp = sbjct_start
                     sbjct_start = sbjct_end
                     sbjct_end = tmp
                     
                     query_string = reverse_complement(query_string)
                     homo_string = homo_string[::-1]
                     sbjct_string = reverse_complement(sbjct_string)
                     strand = 1
                  
                  if cut_off == True:
                     if perc_coverage > 20 :
                        best_hsp = {'evalue': hsp.expect, 'sbjct_header': sbjct_header, 'bit': bit,
                                    'perc_ident': perc_ident, 'sbjct_length':sbjct_length,
                                    'sbjct_start': sbjct_start, 'sbjct_end': sbjct_end,
                                    'gaps': gaps, 'query_string': query_string,
                                    'homo_string': homo_string, 'sbjct_string': sbjct_string,
                                    'contig_name': contig_name, 'query_start': query_start,
                                    'query_end': query_end, 'HSP_length': HSP_length, 'coverage': coverage,
                                    'cal_score': cal_score, 'hit_id': hit_id, 'strand': strand,
                                    'perc_coverage': perc_coverage
                                    }
                  else:
                     best_hsp = {'evalue': hsp.expect, 'sbjct_header': sbjct_header, 'bit': bit,
                                    'perc_ident': perc_ident, 'sbjct_length':sbjct_length,
                                    'sbjct_start': sbjct_start, 'sbjct_end': sbjct_end,
                                    'gaps': gaps, 'query_string': query_string,
                                    'homo_string': homo_string, 'sbjct_string': sbjct_string,
                                    'contig_name': contig_name, 'query_start': query_start,
                                    'query_end': query_end, 'HSP_length': HSP_length, 'coverage': coverage,
                                    'cal_score': cal_score, 'hit_id': hit_id, 'strand': strand,
                                    'perc_coverage': perc_coverage
                                    }
            
            # Saving the result if any
            if best_hsp:
               save = 1
               
               # If there are other gene alignments they are compared
               if gene_results:
                  tmp_gene_split = gene_split
                  tmp_results = gene_results
                  
                  # Compare the hit results
                  save, gene_split, gene_results = compare_results(save, best_hsp, tmp_results, tmp_gene_split)
               
               # If the hit is not overlapping with other hit seqeunces it is kept
               if save == 1:
                  gene_results[hit_id] = best_hsp
            else:
               pass
      
      # If the hit does not cover the entire database reference the missing seqence data are extracted
      for hit_id in list(gene_results):
         hit = gene_results[hit_id]
         
         # Calculate possible split gene coverage
         perc_coverage = hit['perc_coverage']
         
         if hit['sbjct_header'] in gene_split and len(gene_split[hit['sbjct_header']]) > 1:
            # Calculate new length
            new_length = calculate_new_length(gene_split, gene_results, hit)
            hit['split_length'] = new_length
            
            # Calculate new coverage
            perc_coverage = new_length / float(hit['sbjct_length']) * 100
         
         # If the hit is above the minimum length threshold it is kept
         if perc_coverage >= min_cov:
            if hit['coverage'] == 1:
               gene_align_query[db][hit_id] = hit['query_string']
               gene_align_homo[db][hit_id] = hit['homo_string']
               gene_align_sbjct[db][hit_id] = hit['sbjct_string']
            elif hit['coverage'] != 1:
               # Getting the whole database sequence
               for seq_record in SeqIO.parse(db_file, "fasta"):
                  if seq_record.description == hit['sbjct_header']:
                     gene_align_sbjct[db][hit_id] = str(seq_record.seq)
                     break
               
               # Getting the whole contig to extract extra query seqeunce
               contig = ''
               for seq_record in SeqIO.parse(inputfile, "fasta"):
                  if seq_record.description == hit['contig_name']:
                     contig = str(seq_record.seq)
                     break
               
               # Extract extra sequence from query
               query_seq, homo_seq = get_query_align(hit, contig)
               
               # Saving the new alignment sequences
               gene_align_query[db][hit_id] = query_seq
               gene_align_homo[db][hit_id] = homo_seq
         
         else:
            del gene_results[hit_id]
            if hit['sbjct_header'] in gene_split:
               del gene_split[hit['sbjct_header']]
      
      # Save the database result
      if gene_results:
         results[db] = gene_results
      else:
         results[db] = "No hit found"
   return (results, gene_align_query, gene_align_homo, gene_align_sbjct)