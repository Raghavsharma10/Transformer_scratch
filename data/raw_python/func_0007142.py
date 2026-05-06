def get_query_align(hit, contig):
      """
         Function for extracting extra seqeunce data to the query
         alignment if the full reference length are not covered
      """

      # Getting data needed to extract sequences
      query_seq = hit['query_string']
      homo_seq = hit['homo_string']
      sbjct_start = int(hit['sbjct_start'])
      sbjct_end = int(hit['sbjct_end'])
      query_start = int(hit['query_start'])
      query_end = int(hit['query_end'])
      length = int(hit['sbjct_length'])

      # If the alignment doesn't start at the first position data is
      # added to the begnning
      if sbjct_start != 1:
         missing = sbjct_start - 1

         if(query_start >= missing and hit['strand'] != 1
            or hit['strand'] == 1 and missing <= (len(contig) - query_end)):

            # Getting the query sequence.
            # If the the hit is on the other strand the characters
            # are reversed.
            if hit['strand'] == 1:
               start_pos = query_end
               end_pos = query_end + missing
               chars = contig[start_pos:end_pos]
               chars = Blaster.reversecomplement(chars)
            else:
               start_pos = query_start - missing - 1
               end_pos = query_start - 1
               chars = contig[start_pos:end_pos]

            query_seq = chars + str(query_seq)
         else:
            # Getting the query sequence.
            # If the the hit is on the other strand the characters
            # are reversed.
            if hit['strand'] == 1:
               if query_end == len(contig):
                  query_seq = "-" * missing + str(query_seq)
               else:
                  start_pos = query_end
                  chars = contig[start_pos:]
                  chars = Blaster.reversecomplement(chars)

                  query_seq = ("-" * (missing - len(chars))
                               + chars + str(query_seq))
            elif query_start < 3:
               query_seq = "-" * missing + str(query_seq)
            else:
               end_pos = query_start - 2
               chars = contig[0:end_pos]

               query_seq = ("-" * (missing - len(chars))
                            + chars + str(query_seq))

         # Adding to the homo sequence
         spaces = " " * missing
         homo_seq = str(spaces) + str(homo_seq)

      # If the alignment dosen't end and the last position data is
      # added to the end
      if sbjct_end < length:
         missing = length - sbjct_end

         if(missing <= (len(contig) - query_end) and hit['strand'] != 1
            or hit['strand'] == 1 and query_start >= missing):

            # Getting the query sequence.
            # If the the hit is on the other strand the characters
            # are reversed.
            if hit['strand'] == 1:
               start_pos = query_start - missing - 1
               end_pos = query_start - 1
               chars = contig[start_pos:end_pos]
               chars = Blaster.reversecomplement(chars)
            else:
               start_pos = query_end
               end_pos = query_end + missing
               chars = contig[start_pos:end_pos]

            query_seq = query_seq + chars
         else:
            # If the hit is on the other strand the characters are reversed
            if hit['strand'] == 1:
               if query_start < 3:
                  query_seq = query_seq + "-" * missing
               else:
                  end_pos = query_start - 2
                  chars = contig[0:end_pos]
                  chars = Blaster.reversecomplement(chars)

                  query_seq = (query_seq
                               + chars + "-" * (missing - len(chars)))
            elif query_end == len(contig):
               query_seq = query_seq + "-" * missing
            else:
               start_pos = query_end
               chars = contig[start_pos:]

               query_seq = query_seq + chars + "-" * (missing - len(chars))

         # Adding to the homo sequence
         spaces = " " * int(missing)
         homo_seq = str(homo_seq) + str(spaces)

      return query_seq, homo_seq