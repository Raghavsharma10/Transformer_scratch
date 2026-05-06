def inputs_from_dataframe(df, delays=(1, 2, 3), inputs=(1, 2, -1), outputs=None, normalize=True, verbosity=1):
    """ Build a sequence of vectors suitable for "activation" by a neural net

    Identical to `dataset_from_dataframe`, except that only the input vectors are
    returned (not a full DataSet instance) and default values for 2 arguments are changed:
        outputs: None

    And only the input vectors are return
    """
    ds = input_dataset_from_dataframe(df=df, delays=delays, inputs=inputs, outputs=outputs,
                                      normalize=normalize, verbosity=verbosity)
    return ds['input']