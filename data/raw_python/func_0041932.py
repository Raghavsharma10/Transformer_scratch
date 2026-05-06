def cd(doi):
    """
    cd to directory of interest(doi)

    a doi can be:

    herbert - the container named "herbert"
    sdox:dev - product "website" located in container "herbert"
    :param doi:
    :return:
    """

    parts = doi.split(':')

    if len(parts) == 2:
        container_name, product_name = parts[0], parts[1]
    elif len(parts) == 1 and os.environ.get('CONTAINER_NAME'):
        # interpret poi as product name if already zapped into a product in order
        # to enable simply switching products by doing ape zap prod.
        product_name = parts[0]
        container_name = os.environ.get('CONTAINER_NAME')
    else:
        print('unable to parse context - format: <container_name>:<product_name>')
        sys.exit(1)

    if container_name not in tasks.get_containers():
        print('No such container')
    else:
        if product_name:
            if product_name not in tasks.get_products(container_name):
                print('No such product')
            else:
                print(tasks.conf.SOURCE_HEADER)
                print('cd ' + tasks.get_product_dir(container_name, product_name))
        else:
            print(tasks.conf.SOURCE_HEADER)
            print('cd ' + tasks.get_container_dir(container_name))