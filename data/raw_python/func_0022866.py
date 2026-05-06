def surface(self, zdata, **kwargs):
        """Show a 3D surface plot.

        Extra keyword arguments are passed to `SurfacePlot()`.

        Parameters
        ----------
        zdata : array-like
            A 2D array of the surface Z values.

        """
        self._configure_3d()
        surf = scene.SurfacePlot(z=zdata, **kwargs)
        self.view.add(surf)
        self.view.camera.set_range()
        return surf