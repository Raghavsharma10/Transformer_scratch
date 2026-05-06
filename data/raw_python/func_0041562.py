def naive(gold_schemes):
    """find naive baseline (most common scheme of a given length)?"""
    scheme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'schemes.json')
    with open(scheme_path, 'r') as f:
        dist = json.loads(f.read())
    best_schemes = {}
    for i in dist.keys():
        if not dist[i]:
            continue
        best_schemes[int(i)] = tuple(int(j) for j in (max(dist[i], key=lambda x: x[1])[0]).split())

    naive_schemes = []
    for g in gold_schemes:
        naive_schemes.append(best_schemes[len(g)])
    return naive_schemes