def draw_patch(self, ax, patch, force_trans=None):
        """Process a matplotlib patch object and call renderer.draw_path"""
        vertices, pathcodes = utils.SVG_path(patch.get_path())
        transform = patch.get_transform()
        coordinates, vertices = self.process_transform(transform,
                                                       ax, vertices,
                                                       force_trans=force_trans)
        linestyle = utils.get_path_style(patch, fill=patch.get_fill())
        self.renderer.draw_path(data=vertices,
                                coordinates=coordinates,
                                pathcodes=pathcodes,
                                style=linestyle,
                                mplobj=patch)