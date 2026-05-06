def nearly_unique(arr, rel_tol=1e-4, verbose=0):
    '''Heuristic method to return the uniques within some precision in a numpy array'''
    results = np.array([arr[0]])
    for x in arr:
        if np.abs(results - x).min() > rel_tol:
            results = np.append(results, x)
    return results