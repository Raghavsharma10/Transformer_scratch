def _agg_bake(cls, vertices, color, closed=False):
        """
        Bake a list of 2D vertices for rendering them as thick line. Each line
        segment must have its own vertices because of antialias (this means no
        vertex sharing between two adjacent line segments).
        """

        n = len(vertices)
        P = np.array(vertices).reshape(n, 2).astype(float)
        idx = np.arange(n)  # used to eventually tile the color array

        dx, dy = P[0] - P[-1]
        d = np.sqrt(dx*dx+dy*dy)

        # If closed, make sure first vertex = last vertex (+/- epsilon=1e-10)
        if closed and d > 1e-10:
            P = np.append(P, P[0]).reshape(n+1, 2)
            idx = np.append(idx, idx[-1])
            n += 1

        V = np.zeros(len(P), dtype=cls._agg_vtype)
        V['a_position'] = P

        # Tangents & norms
        T = P[1:] - P[:-1]

        N = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
        # T /= N.reshape(len(T),1)
        V['a_tangents'][+1:, :2] = T
        V['a_tangents'][0, :2] = T[-1] if closed else T[0]
        V['a_tangents'][:-1, 2:] = T
        V['a_tangents'][-1, 2:] = T[0] if closed else T[-1]

        # Angles
        T1 = V['a_tangents'][:, :2]
        T2 = V['a_tangents'][:, 2:]
        A = np.arctan2(T1[:, 0]*T2[:, 1]-T1[:, 1]*T2[:, 0],
                       T1[:, 0]*T2[:, 0]+T1[:, 1]*T2[:, 1])
        V['a_angles'][:-1, 0] = A[:-1]
        V['a_angles'][:-1, 1] = A[+1:]

        # Segment
        L = np.cumsum(N)
        V['a_segment'][+1:, 0] = L
        V['a_segment'][:-1, 1] = L
        # V['a_lengths'][:,2] = L[-1]

        # Step 1: A -- B -- C  =>  A -- B, B' -- C
        V = np.repeat(V, 2, axis=0)[1:-1]
        V['a_segment'][1:] = V['a_segment'][:-1]
        V['a_angles'][1:] = V['a_angles'][:-1]
        V['a_texcoord'][0::2] = -1
        V['a_texcoord'][1::2] = +1
        idx = np.repeat(idx, 2)[1:-1]

        # Step 2: A -- B, B' -- C  -> A0/A1 -- B0/B1, B'0/B'1 -- C0/C1
        V = np.repeat(V, 2, axis=0)
        V['a_texcoord'][0::2, 1] = -1
        V['a_texcoord'][1::2, 1] = +1
        idx = np.repeat(idx, 2)

        I = np.resize(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32),
                      (n-1)*(2*3))
        I += np.repeat(4*np.arange(n-1, dtype=np.uint32), 6)

        # Length
        V['alength'] = L[-1] * np.ones(len(V))

        # Color
        if color.ndim == 1:
            color = np.tile(color, (len(V), 1))
        elif color.ndim == 2 and len(color) == n:
            color = color[idx]
        else:
            raise ValueError('Color length %s does not match number of '
                             'vertices %s' % (len(color), n))
        V['color'] = color

        return V, I