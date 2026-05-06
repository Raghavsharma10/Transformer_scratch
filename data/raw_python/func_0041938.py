def config_to_equation(poi=None):
    """
    Generates a product.equation file for the given product name.
    It generates it from the <product_name>.config file in the products folder.
    For that you need to have your project imported to featureIDE and set the correct settings.
    """
    from . import utils

    container_dir, product_name = tasks.get_poi_tuple(poi=poi)
    info_object = utils.get_feature_ide_paths(container_dir, product_name)
    feature_list = list()

    try:
        print('*** Processing ', info_object.config_file_path)
        with open(info_object.config_file_path, 'r') as config_file:

            config_file = config_file.readlines()
            for line in config_file:
                # in FeatureIDE we cant use '.' for the paths to sub-features so we used '__'
                # e.g. django_productline__features__development
                if len(line.split('__')) <= 2:
                    line = line
                else:
                    line = line.replace('__', '.')

                if line.startswith('abstract_'):
                    # we skipp abstract features; this is a special case as featureIDE does not work with abstract
                    # sub trees / leafs.
                    line = ''

                feature_list.append(line)
    except IOError:
        print('{} does not exist. Make sure your config file exists.'.format(info_object.config_file_path))

    feature_list = tasks.get_ordered_feature_list(info_object, feature_list)

    try:
        with open(info_object.equation_file_path, 'w') as eq_file:
            eq_file.writelines(feature_list)
        print('*** Successfully generated product.equation')
    except IOError:
        print('product.equation file not found. Please make sure you have a valid product.equation in your chosen product')

    # finally performing the validation of the product equation
    tasks.validate_product_equation()