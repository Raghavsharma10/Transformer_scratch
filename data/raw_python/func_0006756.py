def compare_results(save, best_hsp, tmp_results, tmp_gene_split):
   ''' Function for comparing hits and saving only the best hit '''
   # Get data for comparison
   hit_id = best_hsp['hit_id']
   new_start_query = best_hsp['query_start']
   new_end_query = best_hsp['query_end']
   new_start_sbjct = int(best_hsp['sbjct_start'])
   new_end_sbjct = int(best_hsp['sbjct_end'])
   new_score = best_hsp['cal_score']
   new_db_hit = best_hsp['sbjct_header']
   new_contig = best_hsp['contig_name']
   new_HSP = best_hsp['HSP_length']
   
   # See if the best HSP fragment overlap with another allignment and keep the
   # allignment with the highest score - if the new fragment is not providing new seqeunce
   for hit in list(tmp_results):
      hit_data = tmp_results[hit]
      old_start_query = hit_data['query_start']
      old_end_query = hit_data['query_end']
      old_start_sbjct = int(hit_data['sbjct_start'])
      old_end_sbjct = int(hit_data['sbjct_end'])
      old_score = hit_data['cal_score']
      old_db_hit = hit_data['sbjct_header']
      old_contig = hit_data['contig_name']
      old_HSP = hit_data['HSP_length']
      
      remove_old = 0
      
      # If they align to the same gene in the database they are compared
      if new_db_hit == old_db_hit:
         # If the hit provids additional sequence it is kept and the new coverage is saved
         # otherwise the one with the highest score is kept
         if new_start_sbjct < (old_start_sbjct) or new_end_sbjct > (old_end_sbjct):
            # Save the hits as splitted
            tmp_gene_split[old_db_hit][hit_id] = 1
            if not hit in tmp_gene_split[old_db_hit]:
               tmp_gene_split[old_db_hit][hit] = 1
         else:
            if new_score > old_score:
               # Set to remove old hit
               remove_old = 1
               
               # Save a split if the new hit still creats one
               if new_db_hit in tmp_gene_split and not hit_id in tmp_gene_split[new_db_hit]:
                  tmp_gene_split[new_db_hit][hit_id] = 1
            else:
               save = 0
               
               # If the old and new hit is not identical the possible saved gene split for the new hit is removed
               if hit_id != hit:
                  if new_db_hit in tmp_gene_split and hit_id in tmp_gene_split[new_db_hit]:
                     del tmp_gene_split[new_db_hit][hit_id]
               break
      
      # If the hits comes form the same part of the contig sequnce but match different genes only the best hit is kept
      if new_contig == old_contig:
         # if the two hits cover the exact same place on the contig only
         # the percentage of identity is compared
         if old_start_query == new_start_query and old_end_query == new_end_query:
            if best_hsp['perc_ident'] > hit_data['perc_ident']:
               
               # Set to remove old hit
               remove_old = 1
               
               # Save a split if the new hit still creats one
               if new_db_hit in tmp_gene_split and not hit_id in tmp_gene_split[new_db_hit]:
                  tmp_gene_split[new_db_hit][hit_id] = 1
            
            elif best_hsp['perc_ident'] == hit_data['perc_ident']:
               # Save both
               
               # Save a split if the new hit still creats one
               if new_db_hit in tmp_gene_split and not hit_id in tmp_gene_split[new_db_hit]:
                  tmp_gene_split[new_db_hit][hit_id] = 1
            else:
               save = 0
                # Remove new gene from gene split if present
               if new_db_hit in tmp_gene_split and hit_id in tmp_gene_split[new_db_hit]:
                  del tmp_gene_split[new_db_hit][hit_id]
               break
         
         elif (max(old_end_query, new_end_query) - min(old_start_query, new_start_query)) <= ((old_end_query - old_start_query) + (new_end_query - new_start_query)):
            if new_score > old_score:
               # Set to remove old gene
               remove_old = 1
               
               # Save a split if the new hit still creats one
               if new_db_hit in tmp_gene_split and not hit_id in tmp_gene_split[new_db_hit]:
                  tmp_gene_split[new_db_hit][hit_id] = 1
            
            elif new_score == old_score:
               # If both genes are completly covered the longest hit is chosen
               if int(best_hsp['perc_coverage']) == 100 and int(hit_data['perc_coverage']) == 100 and new_HSP > old_HSP:
                  # Set to remove old gene
                  remove_old = 1
               
               # Save a split if the new hit creats one - both hits are saved
               if new_db_hit in tmp_gene_split and not hit_id in tmp_gene_split[new_db_hit]:
                  tmp_gene_split[new_db_hit][hit_id] = 1
            else:
               # Remove new gene from gene split if present
               if new_db_hit in tmp_gene_split and hit_id in tmp_gene_split[new_db_hit]:
                  del tmp_gene_split[new_db_hit][hit_id]
               
               save = 0
               break
      
      # Remove old hit if new hit is better
      if remove_old == 1:
         del tmp_results[hit]
         # Remove gene from gene split if present
         if old_db_hit in tmp_gene_split and hit in tmp_gene_split[old_db_hit]:
            del tmp_gene_split[old_db_hit][hit]
   
   return save, tmp_gene_split, tmp_results