def weight_matrices(nn):
    """ Extract list of weight matrices from a Network, Layer (module), Trainer, Connection or other pybrain object"""

    if isinstance(nn, ndarray):
        return nn

    try:
        return weight_matrices(nn.connections)
    except:
        pass

    try:
        return weight_matrices(nn.module)
    except:
        pass

    # Network objects are ParameterContainer's too, but won't reshape into a single matrix,
    # so this must come after try nn.connections
    if isinstance(nn, (ParameterContainer, Connection)):
        return reshape(nn.params, (nn.outdim, nn.indim))

    if isinstance(nn, basestring):
        try:
            fn = nn
            nn = NetworkReader(fn, newfile=False)
            return weight_matrices(nn.readFrom(fn))
        except:
            pass
    # FIXME: what does NetworkReader output? (Module? Layer?) need to handle it's type here

    try:
        return [weight_matrices(v) for (k, v) in nn.iteritems()]
    except:
        try:
            connections = nn.module.connections.values()
            nn = []
            for conlist in connections:
                nn += conlist
            return weight_matrices(nn)
        except:
            return [weight_matrices(v) for v in nn]