def _validate_annotations(annotations, label, add_error):
    """Check that the given service or machine annotations are valid.

    Use the given label (e.g. "machine X" or "service Y") to describe
    possible errors.
    Use the given add_error callable to register validation error.
    """
    if annotations is None:
        return
    if not isdict(annotations):
        add_error('{} has invalid annotations {}'.format(label, annotations))
        return
    # Check that all the annotations keys are strings.
    if not all(map(isstring, annotations)):
        add_error(
            '{} has invalid annotations: keys must be strings'.format(label))