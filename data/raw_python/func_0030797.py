def _do_read_config(self, config_file, pommanipext):
        """Reads config for a single job defined by section."""
        parser = InterpolationConfigParser()
        dataset = parser.read(config_file)
        if config_file not in dataset:
            raise IOError("Config file %s not found." % config_file)
        if parser.has_option('common','include'):
            include = parser.get('common', 'include')
            if include is not "":
                sections_ = self.read_and_load(include)
                for section_ in sections_:
                    if parser.has_section(section_):
                        raise DuplicateSectionError( "The config section [%s] is existed in %s and include %s cfg file" % ( section_, config_file, re.split("\\s+", include.strip())[1]))
                parser._sections.update(sections_)

        pom_manipulator_config = {}
        common_section = {}
        package_configs = {}

        if pommanipext and pommanipext != '' and pommanipext != 'None': #TODO ref: remove none check, it is passed over cmd line in jenkins build
            parse_pom_manipulator_ext(pom_manipulator_config, parser, pommanipext)

        if not parser.has_section('common'):
            logging.error('Mandatory common section missing from configuration file.')
            raise NoSectionError('Mandatory common section missing from configuration file.')
        common_section['tag'] = parser.get('common', 'tag')
        common_section['target'] = parser.get('common', 'target')
        common_section['jobprefix'] = parser.get('common', 'jobprefix')
        common_section['jobciprefix'] = parser.get('common', 'jobciprefix')
        common_section['jobjdk'] = parser.get('common', 'jobjdk')
        if parser.has_option('common', 'mvnver'):
            common_section['mvnver'] = parser.get('common', 'mvnver')
        if parser.has_option('common', 'skiptests'):
            common_section['skiptests'] = parser.get('common', 'skiptests')
        if parser.has_option('common', 'base'):
            common_section['base'] = parser.get('common', 'base')
        if parser.has_option('common', 'citemplate'):
            common_section['citemplate'] = parser.get('common', 'citemplate')
        if parser.has_option('common', 'jenkinstemplate'):
            common_section['jenkinstemplate'] = parser.get('common', 'jenkinstemplate')
        if parser.has_option('common', 'product_name'):
            common_section['product_name'] = parser.get('common', 'product_name')

        if parser.has_option('common', 'include'):
            common_section['include'] = parser.get('common', 'include')

        common_section['jobfailureemail'] = parser.get('common', 'jobfailureemail')

        config_dir = utils.get_dir(config_file)

        #Jira
        if parser.has_option('common', 'shared_config') and parser.get('common', 'shared_config') is not "":
            parse_shared_config(common_section, config_dir, parser)

        common_section['jobtimeout'] = parser.getint('common', 'jobtimeout')

        common_section['options'] = {}
        # If the configuration file has global properties insert these into the common properties map.
        # These may be overridden later by particular properties.
        if parser.has_option('common', 'globalproperties'):
            common_section['options']['properties'] = dict(x.strip().split('=') for x in parser.get('common', 'globalproperties').replace(",\n", ",").split(','))
        else:
            # Always ensure properties has a valid dictionary so code below doesn't need multiple checks.
            common_section['options']['properties'] = {}
        # The same for global profiles
        if parser.has_option('common', 'globalprofiles'):
            common_section['options']['profiles'] = [x.strip() for x in parser.get('common', 'globalprofiles').split(',')]
        else:
            # Always ensure profiles has a valid list so code below doesn't need multiple checks.
            common_section['options']['profiles'] = []

        if os.path.dirname(config_file):
            config_path = os.path.dirname(config_file)
        else:
            config_path = os.getcwd()
        logging.info("Configuration file is %s and path %s", os.path.basename(config_file), config_path)

        for section in parser.sections():
            config_type = self.read_config_type(parser, section)
            if section == 'common' or config_type == "bom-builder-meta":
                logging.debug ('Skipping section due to meta-type %s', section)
                continue

            self._do_read_section(config_path, os.path.basename(config_file), package_configs, parser, section)

        return (common_section, package_configs, pom_manipulator_config)