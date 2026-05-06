def cigar2query(template, cigar):
   ''' Generate query sequence from the template and extended cigar annotation
   
   USAGE:
      >>> template = 'CGATCGATAAATAGAGTAGGAATAGCA'
      >>> cigar = ':6-ata:10+gtc:4*at:3'
      >>> cigar2query(template, cigar) == 'CGATCGAATAGAGTAGGTCGAATtGCA'.upper()
      True
   '''
   query = []
   entries = ['+','-','*',':']
   number = list(map(str,range(10)))
   cigar_length = len(cigar)
   num = []
   entry = None
   pos = 0
   i = 0
   while i < cigar_length:
      if cigar[i] in entries:
         # New entry
         if entry == ':':
            old_pos = pos
            pos += int(''.join(num))
            query.append(template[old_pos:pos])
            num = []
         entry = cigar[i]
         if entry == '*':
            i += 2
            query.append(cigar[i])
            pos += 1
      elif cigar[i] in number:
         num.append(cigar[i])
      elif entry == '-':
         pos += 1
      elif entry == '+':
         query.append(cigar[i])
      i += 1
   
   if entry == ':':
      old_pos = pos
      pos += int(''.join(num))
      query.append(template[old_pos:pos])
   
   return ''.join(query).upper()