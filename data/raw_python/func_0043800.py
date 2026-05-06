def gen_sites(path):
    " Seek sites by path. "

    for root, _, _ in walklevel(path, 2):
        try:
            yield Site(root)
        except AssertionError:
            continue