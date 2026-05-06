def dataset_from_dataframe(df, delays=(1, 2, 3), inputs=(1, 2, -1), outputs=(-1,), normalize=False, verbosity=1):
    """Compose a pybrain.dataset from a pandas DataFrame

    Arguments:
      delays (list of int): sample delays to use for the input tapped delay line
        Positive and negative values are treated the same as sample counts into the past.
        default: [1, 2, 3], in z-transform notation: z^-1 + z^-2 + z^-3
      inputs (list of int or list of str): column indices or labels for the inputs
      outputs (list of int or list of str): column indices or labels for the outputs
      normalize (bool): whether to divide each input to be normally distributed about 0 with std 1

    Returns:
      3-tuple: tuple(dataset, list of means, list of stds)
        means and stds allow normalization of new inputs and denormalization of the outputs

    TODO:

        Detect categorical variables with low dimensionality and split into separate bits
            Vowpel Wabbit hashes strings into an int?
        Detect ordinal variables and convert to continuous int sequence
        SEE: http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    """
    if isinstance(delays, int):
        if delays:
            delays = range(1, delays + 1)
        else:
            delays = [0]
    delays = np.abs(np.array([int(i) for i in delays]))
    inputs = [df.columns[int(inp)] if isinstance(inp, (float, int)) else str(inp) for inp in inputs]
    outputs = [df.columns[int(out)] if isinstance(out, (float, int)) else str(out) for out in (outputs or [])]

    inputs = [fuzzy_get(df.columns, i) for i in inputs]
    outputs = [fuzzy_get(df.columns, o) for o in outputs]

    N_inp = len(inputs)
    N_out = len(outputs)

    inp_outs = inputs + outputs
    if verbosity > 0:
        print("inputs: {}\noutputs: {}\ndelays: {}\n".format(inputs, outputs, delays))
    means, stds = np.zeros(len(inp_outs)), np.ones(len(inp_outs))
    if normalize:
        means, stds = df[inp_outs].mean(), df[inp_outs].std()

    if normalize and verbosity > 0:
        print("Input mean values (used to normalize input biases): {}".format(means[:N_inp]))
        print("Output mean values (used to normalize output biases): {}".format(means[N_inp:]))
    ds = pb.datasets.SupervisedDataSet(N_inp * len(delays), N_out)
    if verbosity > 0:
        print("Dataset dimensions are {}x{}x{} (records x indim x outdim) for {} delays, {} inputs, {} outputs".format(
              len(df), ds.indim, ds.outdim, len(delays), len(inputs), len(outputs)))
    # FIXME: normalize the whole matrix at once and add it quickly rather than one sample at a time
    if delays == np.array([0]) and not normalize:
        if verbosity > 0:
            print("No tapped delay lines (delays) were requested, so using undelayed features for the dataset.")
        assert(df[inputs].values.shape[0] == df[outputs].values.shape[0])
        ds.setField('input', df[inputs].values)
        ds.setField('target', df[outputs].values)
        ds.linkFields(['input', 'target'])
        # for inp, outp in zip(df[inputs].values, df[outputs].values):
        #     ds.appendLinked(inp, outp)
        assert(len(ds['input']) == len(ds['target']))
    else:
        for i, out_vec in enumerate(df[outputs].values):
            if verbosity > 0 and i % 100 == 0:
                print("{}%".format(i / .01 / len(df)))
            elif verbosity > 1:
                print('sample[{i}].target={out_vec}'.format(i=i, out_vec=out_vec))
            if i < max(delays):
                continue
            inp_vec = []
            for delay in delays:
                inp_vec += list((df[inputs].values[i - delay] - means[:N_inp]) / stds[:N_inp])
            ds.addSample(inp_vec, (out_vec - means[N_inp:]) / stds[N_inp:])
    if verbosity > 0:
        print("Dataset now has {} samples".format(len(ds)))
    if normalize:
        return ds, means, stds
    else:
        return ds