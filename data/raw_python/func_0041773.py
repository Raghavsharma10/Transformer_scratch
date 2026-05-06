def get_feature_order_constraints(container_dir):
    """
    Returns the feature order constraints dict defined in featuremodel/productline/feature_order.json
    :param container_dir: the container dir.
    :return: dict
    """
    import json

    file_path = os.path.join(container_dir, '_lib/featuremodel/productline/feature_order.json')
    with open(file_path, 'r') as f:
        ordering_constraints = json.loads(f.read())

    return ordering_constraints