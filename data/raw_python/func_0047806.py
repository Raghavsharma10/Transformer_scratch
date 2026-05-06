def variance(arr):
  """variance of the values, must have 2 or more entries.

  :param arr: list of numbers
  :type arr: number[] a number array
  :return: variance
  :rtype: float

  """
  avg = average(arr)
  return sum([(float(x)-avg)**2 for x in arr])/float(len(arr)-1)