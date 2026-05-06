def _load_yaml_file(yaml_file):
        """ load yaml file and check file content format
        """
        with io.open(yaml_file, 'r', encoding='utf-8') as stream:
            yaml_content = yaml.load(stream)
            FileUtils._check_format(yaml_file, yaml_content)
            return yaml_content