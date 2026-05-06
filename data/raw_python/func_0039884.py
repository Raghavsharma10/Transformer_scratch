def main():
    """
    Entry point when used via command line.

    Features are given using the environment variable ``PRODUCT_EQUATION``.
    If it is not set, ``PRODUCT_EQUATION_FILENAME`` is tried: if it points
    to an existing equation file that selection is used.

    (if ``APE_PREPEND_FEATURES`` is given, those features are prepended)

    If the list of features is empty, ``ape.EnvironmentIncomplete`` is raised.
    """
    # check APE_PREPEND_FEATURES
    features = os.environ.get('APE_PREPEND_FEATURES', '').split()
    # features can be specified inline in PRODUCT_EQUATION
    inline_features = os.environ.get('PRODUCT_EQUATION', '').split()
    if inline_features:
        # append inline features
        features += inline_features
    else:
        # fallback: features are specified in equation file
        feature_file = os.environ.get('PRODUCT_EQUATION_FILENAME', '')
        if feature_file:
            # append features from equation file
            features += get_features_from_equation_file(feature_file)
        else:
            if not features:
                raise EnvironmentIncomplete(
                    'Error running ape:\n'
                    'Either the PRODUCT_EQUATION or '
                    'PRODUCT_EQUATION_FILENAME environment '
                    'variable needs to be set!'
                )

    # run ape with features selected
    run(sys.argv, features=features)