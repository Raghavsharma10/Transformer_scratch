def _line_for_patches(self, data, chart, renderer, series_name):
        """
        Add a line along the top edge of a Patch in a stacked Area Chart; return
        the new Glyph for addition to HoverTool.

        :param data: original data for the graph
        :type data: dict
        :param chart: Chart to add the line to
        :type chart: bokeh.charts.Chart
        :param renderer: GlyphRenderer containing one Patches glyph, to draw
          the line for
        :type renderer: bokeh.models.renderers.GlyphRenderer
        :param series_name: the data series name this Patches represents
        :type series_name: str
        :return: GlyphRenderer for a Line at the top edge of this Patch
        :rtype: bokeh.models.renderers.GlyphRenderer
        """
        # @TODO this method needs a major refactor
        # get the original x and y values, and color
        xvals = deepcopy(renderer.data_source.data['x_values'][0])
        yvals = deepcopy(renderer.data_source.data['y_values'][0])
        line_color = renderer.glyph.fill_color

        # save original values for logging if needed
        orig_xvals = [x for x in xvals]
        orig_yvals = [y for y in yvals]

        # get a list of the values
        new_xvals = [x for x in xvals]
        new_yvals = [y for y in yvals]

        # so when a Patch is made, the first point is (0,0); trash it
        xvals = new_xvals[1:]
        yvals = new_yvals[1:]

        # then, we can tell the last point in the "top" line because it will be
        # followed by a point with the same x value and a y value of 0.
        last_idx = None
        for idx, val in enumerate(xvals):
            if yvals[idx+1] == 0 and xvals[idx+1] == xvals[idx]:
                last_idx = idx
                break

        if last_idx is None:
            logger.error('Unable to find top line of patch (x_values=%s '
                         'y_values=%s', orig_xvals, orig_yvals)
            return None

        # truncate our values to just what makes up the top line
        xvals = xvals[:last_idx+1]
        yvals = yvals[:last_idx+1]

        # Currently (bokeh 0.12.1) HoverTool won't show the tooltip for the last
        # point in our line. As a hack for this, add a point with the same Y
        # value and an X slightly before it.
        lastx = xvals[-1]
        xvals[-1] = lastx - 1000  # 1000 nanoseconds
        xvals.append(lastx)
        yvals.append(yvals[-1])
        # get the actual download counts from the original data
        download_counts = [
            data[series_name][y] for y in range(0, len(yvals) - 1)
        ]
        download_counts.append(download_counts[-1])

        # create a ColumnDataSource for the new overlay line
        data2 = {
            'x': xvals,  # Date/x values are numpy.datetime64
            'y': yvals,
            # the following are hacks for data that we want in the HoverTool
            # tooltip
            'SeriesName': [series_name for _ in yvals],
            # formatted date
            'FmtDate': [self.datetime64_to_formatted_date(x) for x in xvals],
            # to show the exact value, not where the pointer is
            'Downloads': download_counts
        }
        # set the formatted date for our hacked second-to-last point to the
        # same value as the last point
        data2['FmtDate'][-2] = data2['FmtDate'][-1]

        # create the CloumnDataSource, then the line for it, then the Glyph
        line_ds = ColumnDataSource(data2)
        line = Line(x='x', y='y', line_color=line_color)
        lineglyph = chart.add_glyph(line_ds, line)
        return lineglyph