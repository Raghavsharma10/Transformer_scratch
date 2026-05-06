def retrieve_net(wrnt_path):
    '''Rerieves a dumped Network and generates WordNet instance.

    @Args:
    --
    wrnt_path : file_path to the '.wrnt' network dumped file.

    @returns:
    --
    word_net : dict of Word() entities generated from the input file.
    '''
    # Exception Handling of  wrong format.
    if wrnt_path[-4:] != __WRNT_FORMAT: raise Exception(__WRNG_FORMAT_MSG)
    
    file = open(wrnt_path,'rb')
    # retrieving a network  from .wrnt file.
    network = pickle.load(file)
    file.close()
    # Generating Word() instance dictionary from retrieved network.
    word_net = {}
    for n in network:
        word_net[n[0]] = Word(n[0],None,set([network[i][0] for i in n[1:]]))
    # deleting useless resources, for efficient memory usage.
    del network
    return word_net