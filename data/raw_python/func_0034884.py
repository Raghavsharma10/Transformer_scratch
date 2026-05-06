def show_available_noise_curves(return_curves=True, print_curves=False):
    """List available sensitivity curves

    This function lists the available sensitivity curve strings in noise_curves folder.

    Args:
        return_curves (bool, optional): If True, return a list of curve options.
        print_curves (bool, optional): If True, print each curve option.

    Returns:
        (optional list of str): List of curve options.

    Raises:
        ValueError: Both args are False.

    """
    if return_curves is False and print_curves is False:
        raise ValueError("Both return curves and print_curves are False."
                         + " You will not see the options")
    cfd = os.path.dirname(os.path.abspath(__file__))
    curves = [curve.split('.')[0] for curve in os.listdir(cfd + '/noise_curves/')]
    if print_curves:
        for f in curves:
            print(f)
    if return_curves:
        return curves
    return