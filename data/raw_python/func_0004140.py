def create_from_config(self, config):
        """Create a template object file defined in the config object
        """

        configService = ConfigService()
        template = TemplateService()

        template.output = config["output"]["location"]

        template_file = configService.get_template_from_config(config)
        template.input = os.path.basename(template_file)
        template.env = Environment(loader=FileSystemLoader(os.path.dirname(template_file)))

        return template