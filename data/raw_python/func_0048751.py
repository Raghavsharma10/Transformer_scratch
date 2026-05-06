def set_weights_by_dict(self,weights):
    """input: an array of weights <<txname1> <weight1>> <<txname2> <weight2>>...
       if this does not get set then even weighting will be used

    :param weights: [[tx1,wght1],[tx2,wght2],...[txN,wightN]]
    :type weights: list
    """
    self._weights = []
    txnames = [x.name for x in self._transcriptome.transcripts]
    for txname in txnames:
      if txname in weights:
        self._weights.append(float(weights[txname]))
      else:
        self._weights.append(float(0))
    return