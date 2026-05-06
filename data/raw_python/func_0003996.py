def zmat_to_cart(zmat):
    """Converts a ZMatrix back to cartesian coordinates."""

    numbers = zmat["number"]
    N = len(numbers)
    coordinates = np.zeros((N, 3), float)

    # special cases for the first coordinates
    coordinates[1, 2] = zmat["distance"][1]
    if zmat["rel1"][2] == 1:
        sign = -1
    else:
        sign = 1
    coordinates[2, 2] = zmat["distance"][2]*sign*np.cos(zmat["angle"][2])
    coordinates[2, 1] = zmat["distance"][2]*sign*np.sin(zmat["angle"][2])
    coordinates[2] += coordinates[2-zmat["rel1"][2]]

    ref0 = 3
    for (number, distance, rel1, angle, rel2, dihed, rel3) in zmat[3:]:
        ref1 = ref0 - rel1
        ref2 = ref0 - rel2
        ref3 = ref0 - rel3
        if ref1 < 0: ref1 = 0
        if ref2 < 0: ref2 = 0
        if ref3 < 0: ref3 = 0
        # define frame axes
        origin = coordinates[ref1]
        new_z = coordinates[ref2] - origin
        norm_z = np.linalg.norm(new_z)
        if norm_z < 1e-15:
            new_z = np.array([0, 0, 1], float)
        else:
            new_z /= np.linalg.norm(new_z)
        new_x = coordinates[ref3] - origin
        new_x -= np.dot(new_x, new_z)*new_z
        norm_x = np.linalg.norm(new_x)
        if norm_x < 1e-15:
            new_x = random_orthonormal(new_z)
        else:
            new_x /= np.linalg.norm(new_x)
        # we must make our axes frame left handed due to the poor IUPAC
        # definition of the sign of a dihedral angle.
        new_y = -np.cross(new_z, new_x)

        # coordinates of new atom:
        x = distance*np.cos(dihed)*np.sin(angle)
        y = distance*np.sin(dihed)*np.sin(angle)
        z = distance*np.cos(angle)
        coordinates[ref0] = origin + x*new_x + y*new_y + z*new_z
        # loop
        ref0 += 1

    return numbers, coordinates