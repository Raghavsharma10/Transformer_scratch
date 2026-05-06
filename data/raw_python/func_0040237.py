def _generate_mark_code(rule_name):
    """Generates a two digit string based on a provided string

    Args:
        rule_name (str): A configured rule name 'pytest_mark3'.

    Returns:
        str: A two digit code based on the provided string '03'
    """
    code = ''.join([i for i in str(rule_name) if i.isdigit()])
    code = code.zfill(2)
    return code