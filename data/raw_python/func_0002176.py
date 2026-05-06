def load_from_stream(self, header):
        """Populate the CoverageDataset from the protobuf information."""
        self._unpack_attrs(header.atts)
        self.name = header.name
        self.lon_lat_domain = header.latlonRect
        self.proj_domain = header.projRect
        self.date_range = header.dateRange
        self.type = header.coverageType

        for sys in header.coordSys:
            self.coord_systems[sys.name] = sys

        for trans in header.coordTransforms:
            self.transforms[trans.name] = trans

        for ax in header.coordAxes:
            self.axes[ax.name] = ax

        for cov in header.grids:
            self.grids[cov.name] = cov