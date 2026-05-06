def average(arr):
  """average of the values, must have more than 0 entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: average
  :rtype: float

  """
  if len(arr) == 0:
    sys.stderr.write("ERROR: no content in array to take average\n")
    sys.exit()
  if len(arr) == 1:  return arr[0]
  return float(sum(arr))/float(len(arr))