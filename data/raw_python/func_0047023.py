def sort_header(header_text):
  """sort the chromosomes in a header text"""
  lines = header_text.rstrip().split("\n")
  rlens = {}
  for ln in lines:
    m = re.match('@SQ\tSN:(\S+)\tLN:(\S+)',ln)
    if m:
      rlens[m.group(1)] = m.group(2)
  output = ''
  done_lens = False
  for ln in lines:
    if re.match('@SQ\tSN:',ln):
      if not done_lens:
        done_lens = True
        for chr in sorted(rlens.keys()):
          output += "@SQ\tSN:"+chr+"\tLN:"+str(rlens[chr])+"\n"
    else:
      output += ln.rstrip("\n")+"\n"
  return output