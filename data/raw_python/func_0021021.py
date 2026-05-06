def recursive_path(pack, path):
    """Find paths recursively"""
    matches = []
    for root, _, filenames in os.walk(os.path.join(pack, path)):
        for filename in filenames:
            matches.append(os.path.join(root, filename)[len(pack) + 1:])
    return matches