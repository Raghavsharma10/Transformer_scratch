def _find_flats_edges(self, data, mag, direction):
        """
        Extend flats 1 square downstream
        Flats on the downstream side of the flat might find a valid angle,
        but that doesn't mean that it's a correct angle. We have to find
        these and then set them equal to a flat
        """

        i12 = np.arange(data.size).reshape(data.shape)

        flat = mag == FLAT_ID_INT
        flats, n = spndi.label(flat, structure=FLATS_KERNEL3)
        objs = spndi.find_objects(flats)

        f = flat.ravel()
        d = data.ravel()
        for i, _obj in enumerate(objs):
            region = flats[_obj] == i+1
            I = i12[_obj][region]
            J = get_adjacent_index(I, data.shape, data.size)
            f[J] = d[J] == d[I[0]]

        flat = f.reshape(data.shape)
        return flat