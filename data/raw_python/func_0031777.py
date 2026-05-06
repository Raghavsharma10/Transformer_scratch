def get_F_y(fname='binzegger_connectivity_table.json', y=['p23']): 
    '''
    Extract frequency of occurrences of those cell types that are modeled.
    The data set contains cell types that are not modeled (TCs etc.)
    The returned percentages are renormalized onto modeled cell-types, i.e. they sum up to 1 
    '''
    # Load data from json dictionary
    f = open(fname,'r')
    data = json.load(f)
    f.close()
    
    occurr = []
    for cell_type in y:
        occurr += [data['data'][cell_type]['occurrence']]
    return list(np.array(occurr)/np.sum(occurr))