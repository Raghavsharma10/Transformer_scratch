def info():
    """
    List information about this productive environment
    :return:
    """
    print()
    print('root directory         :', tasks.conf.APE_ROOT)
    print()
    print('active container       :', os.environ.get('CONTAINER_NAME', ''))
    print()
    print('active product         :', os.environ.get('PRODUCT_NAME', ''))
    print()
    print('ape feature selection  :', tasks.FEATURE_SELECTION)
    print()
    print('containers and products:')
    print('-' * 30)
    print()
    for container_name in tasks.get_containers():
        print(container_name)
        for product_name in tasks.get_products(container_name):
            print('    ' + product_name)
    print()