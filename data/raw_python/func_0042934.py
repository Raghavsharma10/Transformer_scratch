def filter_dup(lst, lists):
    '''
    filters lst to only include terms that don't have lower rank in another list
    '''
    max_rank = len(lst) + 1
    dct = to_ranked_dict(lst)
    dicts = [to_ranked_dict(l) for l in lists]
    return [word for word in lst if all(dct[word] < dct2.get(word, max_rank) for dct2 in dicts)]