def parser_from_buffer(cls, fp):
        """Construct YamlParser from a file pointer."""
        yaml = YAML(typ="safe")
        return cls(yaml.load(fp))