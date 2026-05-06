def pull_stream(image):
    """
    Return generator of pull status objects
    """
    return (json.loads(s) for s in _get_docker().pull(image, stream=True))