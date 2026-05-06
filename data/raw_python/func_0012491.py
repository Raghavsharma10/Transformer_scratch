def _generate_html(self):
        """
        Generate the HTML for the specified graphs.

        :return:
        :rtype:
        """
        logger.debug('Generating templated HTML')
        env = Environment(
            loader=PackageLoader('pypi_download_stats', 'templates'),
            extensions=['jinja2.ext.loopcontrols'])
        env.filters['format_date_long'] = filter_format_date_long
        env.filters['format_date_ymd'] = filter_format_date_ymd
        env.filters['data_columns'] = filter_data_columns
        template = env.get_template('base.html')

        logger.debug('Rendering template')
        html = template.render(
            project=self.project_name,
            cache_date=self._stats.as_of_datetime,
            user=getuser(),
            host=platform_node(),
            version=VERSION,
            proj_url=PROJECT_URL,
            graphs=self._graphs,
            graph_keys=self.GRAPH_KEYS,
            resources=Resources(mode='inline').render(),
            badges=self._badges
        )
        logger.debug('Template rendered')
        return html