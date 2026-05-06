def make_mead(config=None, run_build=False, environment=1, suffix="", product_name=None, product_version=None,
              look_up_only="",clean_group=False):
    """
    Create Build group based on Make-Mead configuration file
    :param config: Make Mead config name
    :return:
    """
    ret=make_mead_impl(config, run_build, environment, suffix, product_name, product_version, look_up_only, clean_group)
    if type(ret) == int and ret != 0:
        sys.exit(ret)
    return ret