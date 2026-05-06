def input_dataset_from_dataframe(df, delays=(1, 2, 3), inputs=(1, 2, -1), outputs=None, normalize=True, verbosity=1):
    """ Build a dataset with an empty output/target vector

    Identical to `dataset_from_dataframe`, except that default values for 2 arguments:
        outputs: None
    """
    return dataset_from_dataframe(df=df, delays=delays, inputs=inputs, outputs=outputs,
                                  normalize=normalize, verbosity=verbosity)