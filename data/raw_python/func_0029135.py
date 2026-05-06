def check_hook_mechanism_is_intact(module):
    """Check if the hook configuration is absent or has both register AND deregister.

    :param module:
    :return: True if valid plugin / module.
    """
    result = True
    if check_register_present(module):
        result = not result
    if check_deregister_present(module):
        result = not result
    return result