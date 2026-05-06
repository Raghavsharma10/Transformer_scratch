def generate_base_grid(self, vtk_filename=None):
        """
        Run first step of algorithm. Next step is split_voxels
        :param vtk_filename:
        :return:
        """
        nd, ed, ed_dir = self.gen_grid_fcn(self.data.shape, self.voxelsize)
        self.add_nodes(nd)
        self.add_edges(ed, ed_dir, edge_low_or_high=0)

        if vtk_filename is not None:
            self.write_vtk(vtk_filename)