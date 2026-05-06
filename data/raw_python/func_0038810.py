def from_keras_log(csv_path, output_dir_path, **kwargs):
    """Plot accuracy and loss from a Keras CSV log.

    Args:
        csv_path: The path to the CSV log with the actual data.
        output_dir_path: The path to the directory where the resultings plots
            should end up.
    """
    # automatically get seperator by using Python's CSV parser
    data = pd.read_csv(csv_path, sep=None, engine='python')
    _from_keras_log_format(data, output_dir_path=output_dir_path, **kwargs)