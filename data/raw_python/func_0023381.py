def draw_line(self, ax, line, force_trans=None):
        """Process a matplotlib line and call renderer.draw_line"""
        coordinates, data = self.process_transform(line.get_transform(),
                                                   ax, line.get_xydata(),
                                                   force_trans=force_trans)
        linestyle = utils.get_line_style(line)
        if linestyle['dasharray'] is None:
            linestyle = None
        markerstyle = utils.get_marker_style(line)
        if (markerstyle['marker'] in ['None', 'none', None]
                or markerstyle['markerpath'][0].size == 0):
            markerstyle = None
        label = line.get_label()
        if markerstyle or linestyle:
            self.renderer.draw_marked_line(data=data, coordinates=coordinates,
                                           linestyle=linestyle,
                                           markerstyle=markerstyle,
                                           label=label,
                                           mplobj=line)