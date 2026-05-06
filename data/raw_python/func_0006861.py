def sort_and_distribute(array, splits=2):
   """ Sort an array of strings to groups by alphabetically continuous
       distribution
   """
   if not isinstance(array, (list,tuple)): raise TypeError("array must be a list")
   if not isinstance(splits, int): raise TypeError("splits must be an integer")
   remaining = sorted(array)
   if sys.version_info < (3, 0):
      myrange = xrange(splits)
   else:
      myrange = range(splits)
   groups = [[] for i in myrange]
   while len(remaining) > 0:
      for i in myrange:
         if len(remaining) > 0: groups[i].append(remaining.pop(0))
   return groups