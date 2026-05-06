def generate(self):
        """
        Generate all output types and write to disk.
        """
        logger.info('Generating graphs')
        self._generate_graph(
            'by-version',
            'Downloads by Version',
            self._stats.per_version_data,
            'Version'
        )
        self._generate_graph(
            'by-file-type',
            'Downloads by File Type',
            self._stats.per_file_type_data,
            'File Type'
        )
        self._generate_graph(
            'by-installer',
            'Downloads by Installer',
            self._stats.per_installer_data,
            'Installer'
        )
        self._generate_graph(
            'by-implementation',
            'Downloads by Python Implementation/Version',
            self._stats.per_implementation_data,
            'Implementation/Version'
        )
        self._generate_graph(
            'by-system',
            'Downloads by System Type',
            self._stats.per_system_data,
            'System'
        )
        self._generate_graph(
            'by-country',
            'Downloads by Country',
            self._stats.per_country_data,
            'Country'
        )
        self._generate_graph(
            'by-distro',
            'Downloads by Distro',
            self._stats.per_distro_data,
            'Distro'
        )
        self._generate_badges()
        logger.info('Generating HTML')
        html = self._generate_html()
        html_path = os.path.join(self.output_dir, 'index.html')
        with open(html_path, 'wb') as fh:
            fh.write(html.encode('utf-8'))
        logger.info('HTML report written to %s', html_path)
        logger.info('Writing SVG badges')
        for name, svg in self._badges.items():
            path = os.path.join(self.output_dir, '%s.svg' % name)
            with open(path, 'w') as fh:
                fh.write(svg)
            logger.info('%s badge written to: %s', name, path)