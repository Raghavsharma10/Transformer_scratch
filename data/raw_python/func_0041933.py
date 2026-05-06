def switch(poi):
    """
    Zaps into a specific product specified by switch context to the product of interest(poi)
    A poi is:
        sdox:dev - for product "dev" located in container "sdox"

    If poi does not contain a ":" it is interpreted as product name implying that a product within this
    container is already active. So if this task is called with ape zap prod (and the corresponding container is
    already zapped in), than only the product is switched.

    After the context has been switched to sdox:dev additional commands may be available
    that are relevant to sdox:dev
    :param poi: product of interest, string: <container_name>:<product_name> or <product_name>.
    """

    parts = poi.split(':')
    if len(parts) == 2:
        container_name, product_name = parts
    elif len(parts) == 1 and os.environ.get('CONTAINER_NAME'):
        # interpret poi as product name if already zapped into a product in order
        # to enable simply switching products by doing ape zap prod.
        container_name = os.environ.get('CONTAINER_NAME')
        product_name = parts[0]
    else:
        print('unable to find poi: ', poi)
        sys.exit(1)

    if container_name not in tasks.get_containers():
        raise ContainerNotFound('No such container %s' % container_name)
    elif product_name not in tasks.get_products(container_name):
        raise ProductNotFound('No such product %s' % product_name)
    else:
        print(SWITCH_TEMPLATE.format(
            source_header=tasks.conf.SOURCE_HEADER,
            container_name=container_name,
            product_name=product_name
        ))