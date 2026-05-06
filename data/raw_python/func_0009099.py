def information_coefficient(total1,total2,intersect):
    '''a simple jacaard (information coefficient) to compare two lists of overlaps/diffs
    '''
    total = total1 + total2
    return 2.0*len(intersect) / total