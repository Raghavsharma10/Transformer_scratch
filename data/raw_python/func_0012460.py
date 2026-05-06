def generate_graph(self):
        """
        Generate the graph; return a 2-tuple of strings, script to place in the
        head of the HTML document and div content for the graph itself.

        :return: 2-tuple (script, div)
        :rtype: tuple
        """
        logger.debug('Generating graph for %s', self._graph_id)
        # tools to use
        tools = [
            PanTool(),
            BoxZoomTool(),
            WheelZoomTool(),
            SaveTool(),
            ResetTool(),
            ResizeTool()
        ]

        # generate the stacked area graph
        try:
            g = Area(
                self._data, x='Date', y=self._y_series_names,
                title=self._title, stack=True, xlabel='Date',
                ylabel='Downloads', tools=tools,
                # note the width and height will be set by JavaScript
                plot_height=400, plot_width=800,
                toolbar_location='above', legend=False
            )
        except Exception as ex:
            logger.error("Error generating %s graph", self._graph_id)
            logger.error("Data: %s", self._data)
            logger.error("y=%s", self._y_series_names)
            raise ex

        lines = []
        legend_parts = []
        # add a line at the top of each Patch (stacked area) for hovertool
        for renderer in g.select(GlyphRenderer):
            if not isinstance(renderer.glyph, Patches):
                continue
            series_name = renderer.data_source.data['series'][0]
            logger.debug('Adding line for Patches %s (series: %s)', renderer,
                         series_name)
            line = self._line_for_patches(self._data, g, renderer, series_name)
            if line is not None:
                lines.append(line)
                legend_parts.append((series_name, [line]))

        # add the Hovertool, specifying only our line glyphs
        g.add_tools(
            HoverTool(
                tooltips=[
                    (self._y_name, '@SeriesName'),
                    ('Date', '@FmtDate'),
                    ('Downloads', '@Downloads'),
                ],
                renderers=lines,
                line_policy='nearest'
            )
        )

        # legend outside chart area
        legend = Legend(legends=legend_parts, location=(0, 0))
        g.add_layout(legend, 'right')
        return components(g)