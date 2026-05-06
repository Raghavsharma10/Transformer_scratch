def clone_source(self):
        " Clone source and prepare templates "

        print_header('Clone src: %s' % self.src, '-')

        # Get source
        source_dir = self._get_source()

        # Append settings from source
        self.read(op.join(source_dir, settings.CFGNAME))

        self.templates += (self.args.template or self.template).split(',')
        self.templates = OrderedSet(self._gen_templates(self.templates))
        self['template'] = ','.join(str(x[0]) for x in self.templates)

        print_header('Deploy templates: %s' % self.template, sep='-')
        with open(op.join(self.deploy_dir, settings.TPLNAME), 'w') as f:
            f.write(self.template)

        with open(op.join(self.deploy_dir, settings.CFGNAME), 'w') as f:
            self['deploy_dir'], tmp_dir = self.target_dir, self.deploy_dir
            self.write(f)
            self['deploy_dir'] = tmp_dir

        # Create site
        site = Site(self.deploy_dir)

        # Prepare templates
        for template_name, template in self.templates:
            site.paste_template(template_name, template, tmp_dir)

        # Create site
        if self.args.info:
            print_header('Project context', sep='-')
            LOGGER.debug(site.get_info(full=True))
            return None

        # Check requirements
        call('sudo chmod +x %s/*.sh' % self.service_dir)
        site.run_check(service_dir=self.service_dir)

        # Save options
        site.write()

        return site