def with_vtk(plot=True):
    """ Tests VTK interface and mesh repair of Stanford Bunny Mesh """
    mesh = vtki.PolyData(bunny_scan)
    meshfix = pymeshfix.MeshFix(mesh)
    if plot:
        print('Plotting input mesh')
        meshfix.plot()
    meshfix.repair()
    if plot:
        print('Plotting repaired mesh')
        meshfix.plot()

    return meshfix.mesh