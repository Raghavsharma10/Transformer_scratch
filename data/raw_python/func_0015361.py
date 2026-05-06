def assistant_from_yaml(cls, source, y, superassistant, fully_loaded=True,
                            role=settings.DEFAULT_ASSISTANT_ROLE):
        """Constructs instance of YamlAssistant loaded from given structure y, loaded
        from source file source.

        Args:
            source: path to assistant source file
            y: loaded yaml structure
            superassistant: superassistant of this assistant
        Returns:
            YamlAssistant instance constructed from y with source file source
        Raises:
            YamlError: if the assistant is malformed
        """
        # In pre-0.9.0, we required assistant to be a mapping of {name: assistant_attributes}
        # now we allow that, but we also allow omitting the assistant name and putting
        # the attributes to top_level, too.
        name = os.path.splitext(os.path.basename(source))[0]
        yaml_checker.check(source, y)
        assistant = yaml_assistant.YamlAssistant(name, y, source, superassistant,
            fully_loaded=fully_loaded, role=role)

        return assistant