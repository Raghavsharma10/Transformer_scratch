def calculate_new_length(gene_split, gene_results, hit):
   ''' Function for calcualting new length if the gene is split on several
   contigs
   '''
   # Looping over splitted hits and calculate new length
   first = 1
   for split in gene_split[hit['sbjct_header']]:
      new_start = int(gene_results[split]['sbjct_start'])
      new_end = int(gene_results[split]['sbjct_end'])
      
      # Get the frist HSP
      if first == 1:
         new_length = int(gene_results[split]['HSP_length'])
         old_start = new_start
         old_end = new_end
         first = 0
         continue
      if new_start < old_start:
         new_length = new_length + (old_start - new_start)
         old_start = new_start
      
      if new_end > old_end:
         new_length = new_length + (new_end - old_end)
         old_end = new_end
   
   return(new_length)