def knt2mlt(t):
    """Count multiplicities of elements in a sorted list or rank-1 array.

Minimal emulation of MATLAB's ``knt2mlt``.

Parameters:
    t:
        Python list or rank-1 array. Must be sorted!

Returns:
    out
        rank-1 array such that
        out[k] = #{ t[i] == t[k] for i < k }

Example:
    If ``t = [1, 1, 2, 3, 3, 3]``, then ``out = [0, 1, 0, 0, 1, 2]``.

Caveat:
    Requires input to be already sorted (this is not checked).
"""
    t = np.atleast_1d(t)
    if t.ndim > 1:
        raise ValueError("t must be a list or a rank-1 array")

    out   = []
    e     = None
    for k in range(t.shape[0]):
        if t[k] != e:
            e     = t[k]
            count = 0
        else:
            count += 1
        out.append(count)

    return np.array( out )