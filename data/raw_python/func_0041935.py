def get_poi_tuple(poi=None):
    """
    Takes the poi or None and returns the container_dir and the product name either of the passed poi
    (<container_name>: <product_name>) or from os.environ-
    :param poi: optional; <container_name>: <product_name>
    :return: tuple of the container directory and the product name
    """
    if poi:
        parts = poi.split(':')
        if len(parts) == 2:
            container_name, product_name = parts
            if container_name not in tasks.get_containers():
                print('No such container')
                sys.exit(1)
            elif product_name not in tasks.get_products(container_name):
                print('No such product')
                sys.exit(1)
            else:
                container_dir = tasks.get_container_dir(container_name)
        else:
            print('Please check your arguments: --poi <container>:<product>')
            sys.exit(1)
    else:
        container_dir = os.environ.get('CONTAINER_DIR')
        product_name = os.environ.get('PRODUCT_NAME')

    return container_dir, product_name