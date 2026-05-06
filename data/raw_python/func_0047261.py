def get_weighted_random_index(self,weights):
    """Return an index of an array based on the weights
       if a random number between 0 and 1 is less than an index return the lowest index

    :param weights: a list of floats for how to weight each index [w1, w2, ... wN]
    :type weights: list
    :return: index
    :rtype: int
    """
    tot = float(sum([float(x) for x in weights]))
    fracarray = [weights[0]]
    for w in weights[1:]:
      prev = fracarray[-1]
      fracarray.append(w+prev)
    #print fracarray
    rnum = self._random.random()*tot
    #print rnum
    #sys.exit()
    for i in range(len(weights)):
      if rnum < fracarray[i]: return i
    sys.stderr.write("Warning unexpected no random\n")