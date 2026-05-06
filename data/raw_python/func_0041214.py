def build_ann(N_input=None, N_hidden=2, N_output=1, hidden_layer_type='Linear', verbosity=1):
    """Build a neural net with the indicated input, hidden, and outout dimensions

    Arguments:
      params (dict or PyBrainParams namedtuple):
        default: {'N_hidden': 6}
        (this is the only parameter that affects the NN build)

    Returns:
        FeedForwardNetwork with N_input + N_hidden + N_output nodes in 3 layers
    """
    N_input = N_input or 1
    N_output = N_output or 1
    N_hidden = N_hidden or tuple()
    if isinstance(N_hidden, (int, float, basestring)):
        N_hidden = (int(N_hidden),)

    hidden_layer_type = hidden_layer_type or tuple()
    hidden_layer_type = tuplify(normalize_layer_type(hidden_layer_type))

    if verbosity > 0:
        print(N_hidden, ' layers of type ', hidden_layer_type)

    assert(len(N_hidden) == len(hidden_layer_type))
    nn = pb.structure.FeedForwardNetwork()

    # layers
    nn.addInputModule(pb.structure.BiasUnit(name='bias'))
    nn.addInputModule(pb.structure.LinearLayer(N_input, name='input'))
    for i, (Nhid, hidlaytype) in enumerate(zip(N_hidden, hidden_layer_type)):
        Nhid = int(Nhid)
        nn.addModule(hidlaytype(Nhid, name=('hidden-{}'.format(i) if i else 'hidden')))
    nn.addOutputModule(pb.structure.LinearLayer(N_output, name='output'))

    # connections
    nn.addConnection(pb.structure.FullConnection(nn['bias'],  nn['hidden'] if N_hidden else nn['output']))
    nn.addConnection(pb.structure.FullConnection(nn['input'], nn['hidden'] if N_hidden else nn['output']))
    for i, (Nhid, hidlaytype) in enumerate(zip(N_hidden[:-1], hidden_layer_type[:-1])):
        Nhid = int(Nhid)
        nn.addConnection(pb.structure.FullConnection(nn[('hidden-{}'.format(i) if i else 'hidden')],
                         nn['hidden-{}'.format(i + 1)]))
    i = len(N_hidden) - 1
    nn.addConnection(pb.structure.FullConnection(nn['hidden-{}'.format(i) if i else 'hidden'], nn['output']))

    nn.sortModules()
    if FAST:
        try:
            nn.convertToFastNetwork()
        except:
            if verbosity > 0:
                print('Unable to convert slow PyBrain NN to a fast ARAC network...')
    if verbosity > 0:
        print(nn.connections)
    return nn