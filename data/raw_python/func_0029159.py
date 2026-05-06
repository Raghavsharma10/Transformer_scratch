def generate_template(context, config, cloudformation):
    """call cloudformation to generate the template (json format).

    :param context:
    :param config:
    :param cloudformation:
    :return:
    """
    spec = inspect.getargspec(cloudformation.generate_template)[0]
    if len(spec) == 0:
        return cloudformation.generate_template()
    elif spec == ['context', 'config']:
        return cloudformation.generate_template(context, config)
    else:
        raise Exception('Arguments of \'generate_template\' not as expected: %s' % spec)