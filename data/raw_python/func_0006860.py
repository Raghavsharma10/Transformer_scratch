def sort2groups(array, gpat=['_R1','_R2']):
   """ Sort an array of strings to groups by patterns """
   groups = [REGroup(gp) for gp in gpat]
   unmatched = []
   for item in array:
      matched = False
      for m in groups:
         if m.match(item):
            matched = True
            break
      if not matched: unmatched.append(item)
   return [sorted(m.list) for m in groups], sorted(unmatched)