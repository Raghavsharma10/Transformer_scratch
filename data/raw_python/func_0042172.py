def inverse(d):
    """
    reverse the k:v pairs
    """
    output = {}
    for k, v in unwrap(d).items():
        output[v] = output.get(v, [])
        output[v].append(k)
    return output