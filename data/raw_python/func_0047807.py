def N50(arr):
  """N50 often used in assessing denovo assembly.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: N50
  :rtype: float

  """
  if len(arr) == 0:
    sys.stderr.write("ERROR: no content in array to take N50\n")
    sys.exit()
  tot = sum(arr)
  half = float(tot)/float(2)
  cummulative = 0
  for l in sorted(arr):
    cummulative += l
    if float(cummulative) > half: 
      return l
  sys.stderr.write("ERROR: problem finding M50\n")
  sys.exit()