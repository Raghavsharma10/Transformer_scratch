def _retrieve_stack_host_zone_name(awsclient, default_stack_name=None):
    """
    Use service discovery to get the host zone name from the default stack

    :return: Host zone name as string
    """
    global _host_zone_name

    if _host_zone_name is not None:
        return _host_zone_name

    env = get_env()

    if env is None:
        print("Please set environment...")
        # TODO: why is there a sys.exit in library code used by cloudformation!!!
        sys.exit()

    if default_stack_name is None:
        # TODO why 'dp-<env>'? - this should not be hardcoded!
        default_stack_name = 'dp-%s' % env
    default_stack_output = get_outputs_for_stack(awsclient, default_stack_name)

    if HOST_ZONE_NAME__STACK_OUTPUT_NAME not in default_stack_output:
        print("Please debug why default stack '{}' does not contain '{}'...".format(
                default_stack_name,
                HOST_ZONE_NAME__STACK_OUTPUT_NAME,
        ))
        # TODO: why is there a sys.exit in library code used by cloudformation!!!
        sys.exit()

    _host_zone_name = default_stack_output[HOST_ZONE_NAME__STACK_OUTPUT_NAME] + "."
    return _host_zone_name