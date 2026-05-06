def create_pth():
    """
    Create the default PTH file
    :return:
    """
    if prefix == '/usr':
        print("Not creating PTH in real prefix: %s" % prefix)
        return False
    with open(vext_pth, 'w') as f:
        f.write(DEFAULT_PTH_CONTENT)
    return True