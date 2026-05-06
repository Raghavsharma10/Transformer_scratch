def different_random_nt(self,nt):
    global nts
    """generate a random nucleotide change. uniform random.  will never return itself

    :param nt: current nucleotide
    :type nt: char
    :return: new nucleotide
    :rtype: char
    """
    return self._random.choice([x for x in nts if x != nt.upper()])