def yaml_block(self):
        """Lazy load a yaml_block.

        If yaml support is not available,
        there is an error in parsing the yaml block,
        or no yaml is associated with this result,
        ``None`` will be returned.

        :rtype: dict
        """
        if LOAD_YAML and self._yaml_block is not None:
            try:
                yaml_dict = yaml.load(self._yaml_block)
                return yaml_dict
            except yaml.error.YAMLError:
                print("Error parsing yaml block. Check formatting.")
        return None