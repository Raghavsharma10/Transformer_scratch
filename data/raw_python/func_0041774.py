def get_features_from_equation(container_dir, product_name):
    """
    Takes the container dir and the product name and returns the list of features.
    :param container_dir: path of the container dir
    :param product_name: name of the product
    :return: list of strings, each representing one feature
    """
    import featuremonkey
    file_path = os.path.join(container_dir, 'products', product_name, 'product.equation')
    return featuremonkey.get_features_from_equation_file(file_path)