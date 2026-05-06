def pprnt(input, return_data=False):
    """
    Prettier print for nested data

    Args:
        input: Input data
        return_data (bool): Default False. Print outs if False, returns if True.
    Returns:
        None | Pretty formatted text representation of input data.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    import json, re
    result = json.dumps(input, sort_keys=True, indent=4)
    result = re.sub(r'(")(\w*?_id)(":)', r'\1%s%s\2%s\3' % (BOLD, HEADER, ENDC), result)
    result = re.sub(r'(")(\w*?_set)(":)', r'\1%s%s\2%s\3' % (BOLD, HEADER, ENDC), result)
    result = re.sub(r'(\n *?")(\w*?)(":)', r'\1%s%s\2%s\3' % (BOLD, OKGREEN, ENDC), result)
    if not return_data:
        print(result)
    else:
        return result