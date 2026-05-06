def string_to_genomic_range(rstring):
  """ Convert a string to a genomic range

  :param rstring: string representing a genomic range chr1:801-900
  :type rstring:
  :returns: object representing the string
  :rtype: GenomicRange
  """
  m = re.match('([^:]+):(\d+)-(\d+)',rstring)
  if not m: 
    sys.stderr.write("ERROR: problem with range string "+rstring+"\n")
  return GenomicRange(m.group(1),int(m.group(2)),int(m.group(3)))