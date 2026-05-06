def plot(self, show_holes=True):
        """
        Plot the mesh.

        Parameters
        ----------
        show_holes : bool, optional
            Shows boundaries.  Default True
        """
        if show_holes:
            edges = self.mesh.extract_edges(boundary_edges=True,
                                            feature_edges=False,
                                            manifold_edges=False)

            plotter = vtki.Plotter()
            plotter.add_mesh(self.mesh, label='mesh')
            plotter.add_mesh(edges, 'r', label='edges')
            plotter.plot()

        else:
            self.mesh.plot(show_edges=True)