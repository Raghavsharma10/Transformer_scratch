def manifest(self, values, *paths, filename: str = None) -> Dict:
        """Load a manifest file and apply template values
        """
        filename = filename or self.filename(*paths)
        with open(filename, 'r') as fp:
            template = Template(fp.read())
        return yaml.load(template.render(values))