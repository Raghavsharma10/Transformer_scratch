def check_reference(images, reference):
    """
    Ensure the reference matches image dimensions
    """
    if not images.shape[1:] == reference.shape:
        raise Exception('Image shape %s and reference shape %s must match'
                        % (images.shape[1:], reference.shape))
    return reference