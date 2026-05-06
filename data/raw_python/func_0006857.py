def seqs_from_file(filename, exit_on_err=False, return_qual=False):
   """Extract sequences from a file
   
   Name:
      seqs_from_file
   Author(s):
      Martin C F Thomsen
   Date:
      18 Jul 2013
   Description:
      Iterator which extract sequence data from the input file
   Args:
      filename: string which contain a path to the input file
   Supported Formats:
      fasta, fastq
   
   USAGE:
   >>> import os, sys
   >>> # Create fasta test file
   >>> file_content = ('>head1 desc1\nthis_is_seq_1\n>head2 desc2\n'
                       'this_is_seq_2\n>head3 desc3\nthis_is_seq_3\n')
   >>> with open_('test.fsa', 'w') as f: f.write(file_content)
   >>> # Parse and print the fasta file
   >>> for seq, name, desc in SeqsFromFile('test.fsa'):
   ...    print ">%s %s\n%s"%(name, desc, seq)
   ...
   >head1 desc1
   this_is_seq_1
   >head2 desc2
   this_is_seq_2
   >head3 desc3
   this_is_seq_3
   """
   # VALIDATE INPUT
   if not isinstance(filename, str):
      msg = 'Filename has to be a string.'
      if exit_on_err:
         sys.stderr.write('Error: %s\n'%msg)
         sys.exit(1)
      else: raise IOError(msg)
   if not os.path.exists(filename):
      msg = 'File "%s" does not exist.'%filename
      if exit_on_err:
         sys.stderr.write('Error: %s\n'%msg)
         sys.exit(1)
      else: raise IOError(msg)
   
   # EXTRACT DATA
   with open_(filename,"rt") as f:
      query_seq_segments = []
      seq, name, desc, qual = '', '', '', ''
      add_segment = query_seq_segments.append
      for l in f:
         if len(l.strip()) == 0: continue
         #sys.stderr.write("%s\n"%line)
         fields=l.strip().split()
         if l.startswith(">"):
            # FASTA HEADER FOUND
            if query_seq_segments != []:
               # YIELD SEQUENCE AND RESET
               seq = ''.join(query_seq_segments)
               yield (seq, name, desc)
               seq, name, desc = '', '', ''
               del query_seq_segments[:]
            name = fields[0][1:]
            desc = ' '.join(fields[1:])
         
         elif l.startswith("@"):
            # FASTQ HEADER FOUND
            name = fields[0][1:]
            desc = ' '.join(fields[1:])
            try:
               # EXTRACT FASTQ SEQUENCE
               seq  = next(f).strip().split()[0]
               # SKIP SECOND HEADER LINE AND QUALITY SCORES
               l = next(f)
               qual = next(f).strip() # Qualities
            except:
               break
            else:
               # YIELD SEQUENCE AND RESET
               if return_qual:
                  yield (seq, qual, name, desc)
               else:
                  yield (seq, name, desc)
               seq, name, desc, qual = '', '', '', ''
         
         elif len(fields[0])>0:
            # EXTRACT FASTA SEQUENCE
            add_segment(fields[0])
      
      # CHECK FOR LAST FASTA SEQUENCE
      if query_seq_segments != []:
         # YIELD SEQUENCE
         seq = ''.join(query_seq_segments)
         yield (seq, name, desc)