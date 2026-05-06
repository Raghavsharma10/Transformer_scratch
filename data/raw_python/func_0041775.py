def get_feature_ide_paths(container_dir, product_name):
    """
    Takes the container_dir and the product name and returns all relevant paths from the
    feature_order_json to the config_file_path.
    :param container_dir: the full path of the container dir
    :param product_name: the name of the product
    :return: object with divert path attributes
    """
    repo_name = get_repo_name(container_dir)

    class Paths(object):
        feature_order_json = os.path.join(container_dir, '_lib/featuremodel/productline/feature_order.json')
        model_xml_path = os.path.join(container_dir, '_lib/featuremodel/productline/model.xml')
        config_file_path = os.path.join(container_dir, '_lib/featuremodel/productline/products/', repo_name, product_name, 'product.equation.config')
        equation_file_path = os.path.join(container_dir, 'products', product_name, 'product.equation')
        product_spec_path = os.path.join(container_dir, '_lib/featuremodel/productline/products/', repo_name, 'product_spec.json')

    return Paths