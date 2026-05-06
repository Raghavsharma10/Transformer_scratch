def plot(data, output_dir_path='.', width=10, height=8):
    """Create two plots: 1) loss 2) accuracy.
        Args:
            data: Panda dataframe in *the* format.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    plot_accuracy(data, output_dir_path=output_dir_path,
                  width=width, height=height)
    plot_loss(data, output_dir_path, width=width, height=height)