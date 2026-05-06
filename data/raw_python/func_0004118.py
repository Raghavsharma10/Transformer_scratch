def get_template_from_config(self, config):
        """Retrieve a template path from the config object
        """
        if config["output"]["template"] == "default":
            return os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'template',
                'default.html'
            )
        else:
            return os.path.abspath(config["output"]["template"])