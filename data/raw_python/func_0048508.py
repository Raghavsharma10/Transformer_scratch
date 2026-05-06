def random_flip(sequence,rnum=None):
  """Flip a sequence direction with 0.5 probability"""
  randin = rnum
  if not randin: randin = RandomSource()
  if randin.random() < 0.5:
    return rc(sequence)
  return sequence