def get_cube_points(origin, axes, nrep):
    '''Generate the Cartesian coordinates of the points in a cube file

       *Arguemnts:*

       origin
            The cartesian coordinate for the origin of the grid.

       axes
            The 3 by 3 array with the grid spacings as rows.

       nrep
            The number of grid points along each axis.
    '''
    points = np.zeros((nrep[0], nrep[1], nrep[2], 3), float)
    points[:] = origin
    points[:] += np.outer(np.arange(nrep[0], dtype=float), axes[0]).reshape((-1,1,1,3))
    points[:] += np.outer(np.arange(nrep[1], dtype=float), axes[1]).reshape((1,-1,1,3))
    points[:] += np.outer(np.arange(nrep[2], dtype=float), axes[2]).reshape((1,1,-1,3))
    return points