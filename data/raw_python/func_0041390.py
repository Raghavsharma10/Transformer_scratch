def train_weather_predictor(
        location='Portland, OR',
        years=range(2013, 2016,),
        delays=(1, 2, 3),
        inputs=('Min Temperature', 'Max Temperature', 'Min Sea Level Pressure', u'Max Sea Level Pressure', 'WindDirDegrees',),
        outputs=(u'Max TemperatureF',),
        N_hidden=6,
        epochs=30,
        use_cache=False,
        verbosity=2,
        ):
    """Train a neural nerual net to predict the weather for tomorrow based on past weather.

    Builds a linear single hidden layer neural net (multi-dimensional nonlinear regression).
    The dataset is a basic SupervisedDataSet rather than a SequentialDataSet, so the training set
    and the test set are sampled randomly. This means that historical data for one sample (the delayed
    input vector) will likely be used as the target for other samples.

    Uses CSVs scraped from wunderground (without an api key) to get daily weather for the years indicated.

    Arguments:
      location (str): City and state in standard US postal service format: "City, ST"
          alternatively an airport code like "PDX or LAX"
      delays (list of int): sample delays to use for the input tapped delay line.
          Positive and negative values are treated the same as sample counts into the past.
          default: [1, 2, 3], in z-transform notation: z^-1 + z^-2 + z^-3
      years (int or list of int): list of 4-digit years to download weather from wunderground
      inputs (list of int or list of str): column indices or labels for the inputs
      outputs (list of int or list of str): column indices or labels for the outputs

    Returns:
      3-tuple: tuple(dataset, list of means, list of stds)
          means and stds allow normalization of new inputs and denormalization of the outputs

    """
    df = weather.daily(location, years=years, use_cache=use_cache, verbosity=verbosity).sort()
    ds = util.dataset_from_dataframe(df, normalize=False, delays=delays, inputs=inputs, outputs=outputs, verbosity=verbosity)
    nn = util.ann_from_ds(ds, N_hidden=N_hidden, verbosity=verbosity)
    trainer = util.build_trainer(nn, ds=ds, verbosity=verbosity)
    trainer.trainEpochs(epochs)

    columns = []
    for delay in delays:
        columns += [inp + "[-{}]".format(delay) for inp in inputs]
    columns += list(outputs)

    columns += ['Predicted {}'.format(outp) for outp in outputs]
    table = [list(i) + list(t) + list(trainer.module.activate(i)) for i, t in zip(trainer.ds['input'], trainer.ds['target'])]
    df = pd.DataFrame(table, columns=columns, index=df.index[max(delays):])

    #comparison = df[[] + list(outputs)]
    return trainer, df