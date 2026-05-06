def from_config(cls, config, name, section_key="segmenters"):
        """
        Constructs a segmenter from a configuration doc.
        """
        section = config[section_key][name]
        segmenter_class_path = section['class']
        Segmenter = yamlconf.import_module(segmenter_class_path)
        return Segmenter.from_config(config, name, section_key=section_key)