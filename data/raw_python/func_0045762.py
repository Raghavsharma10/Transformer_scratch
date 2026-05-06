def sort_genomic_ranges(rngs):
  """sort multiple ranges"""
  return sorted(rngs, key=lambda x: (x.chr, x.start, x.end))