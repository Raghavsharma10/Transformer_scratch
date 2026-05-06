def split_voxels(self, vtk_filename=None):
        """
        Second step of algorithm
        :return:()
        """
        self.cache = {}
        self.stats["t graph 10"] = time.time() - self.start_time
        self.msi = MultiscaleArray(self.data.shape, block_size=self.nsplit)

        # old implementation
        # idxs = nm.where(self.data)
        # nr, nc = self.data.shape
        # for k, (ir, ic) in enumerate(zip(*idxs)):
        #     ndid = ic + ir * nc
        #     self.split_voxel(ndid, self.nsplit)

        # new_implementation
        # for ndid in np.flatnonzero(self.data):
        #     self.split_voxel(ndid, self.nsplit)

        # even newer implementation
        self.stats["t graph 11"] = time.time() - self.start_time
        for ndid, val in enumerate(self.data.ravel()):
            t_split_start = time.time()
            if val == 0:
                if self.compute_msindex:
                    self.msi.set_block_lowres(ndid, ndid)
                self.stats["t graph low"] += time.time() - t_split_start
            else:
                self.split_voxel(ndid)
                self.stats["t graph high"] += time.time() - t_split_start

        self.stats["t graph 13"] = time.time() - self.start_time
        self.finish()
        if vtk_filename is not None:
            self.write_vtk(vtk_filename)
        self.stats["t graph 14"] = time.time() - self.start_time