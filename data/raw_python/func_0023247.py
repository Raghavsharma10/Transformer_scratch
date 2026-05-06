def merge(a_intervals, b_intervals, op):
        """
        Merge two lists of intervals according to the boolean function op

        ``a_intervals`` and ``b_intervals`` need to be sorted and consistent (no overlapping intervals).
        This operation keeps the resulting interval set consistent.

        Parameters
        ----------
        a_intervals : `~numpy.ndarray`
            A sorted merged list of intervals represented as a N x 2 numpy array
        b_intervals : `~numpy.ndarray`
            A sorted merged list of intervals represented as a N x 2 numpy array
        op : `function`
            Lambda function taking two params and returning the result of the operation between
            these two params.
            Exemple : lambda in_a, in_b: in_a and in_b describes the intersection of ``a_intervals`` and
            ``b_intervals`` whereas lambda in_a, in_b: in_a or in_b describes the union of ``a_intervals`` and
            ``b_intervals``.

        Returns
        -------
        array : `numpy.ndarray`
            a N x 2 numpy containing intervals resulting from the op between ``a_intervals`` and ``b_intervals``.
        """
        a_endpoints = a_intervals.flatten().tolist()
        b_endpoints = b_intervals.flatten().tolist()

        sentinel = max(a_endpoints[-1], b_endpoints[-1]) + 1

        a_endpoints += [sentinel]
        b_endpoints += [sentinel]

        a_index = 0
        b_index = 0

        res = []

        scan = min(a_endpoints[0], b_endpoints[0])
        while scan < sentinel:
            in_a = not ((scan < a_endpoints[a_index]) ^ (a_index % 2))
            in_b = not ((scan < b_endpoints[b_index]) ^ (b_index % 2))
            in_res = op(in_a, in_b)

            if in_res ^ (len(res) % 2):
                res += [scan]
            if scan == a_endpoints[a_index]:
                a_index += 1
            if scan == b_endpoints[b_index]:
                b_index += 1

            scan = min(a_endpoints[a_index], b_endpoints[b_index])

        return np.asarray(res).reshape((-1, 2))