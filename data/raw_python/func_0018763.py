def add_schema_spec(self, spec: Dict[str, Any],
                        fully_qualified_parent_name: str = None) -> Optional[str]:
        """
        Add a schema dictionary to the schema loader. The given schema is stored
        against fully_qualified_parent_name + ITEM_SEPARATOR('.') + schema.name.
        :param spec: Schema specification.
        :param fully_qualified_parent_name: Full qualified name of the parent.
        If None is passed then the schema is stored against the schema name.
        :return: The fully qualified name against which the spec is stored.
        None is returned if the given spec is not a dictionary or the spec does not
        contain a 'name' key.
        """
        if not isinstance(spec, dict) or ATTRIBUTE_NAME not in spec:
            return None

        name = spec[ATTRIBUTE_NAME]
        fully_qualified_name = name if fully_qualified_parent_name is None else self.get_fully_qualified_name(
            fully_qualified_parent_name, name)

        # Ensure that basic validation for each spec part is done before it is added to spec cache
        if isinstance(spec, dict):
            self._error_cache.add(
                validate_required_attributes(fully_qualified_name, spec, ATTRIBUTE_NAME,
                                             ATTRIBUTE_TYPE))
            if ATTRIBUTE_TYPE in spec and not Type.contains(spec[ATTRIBUTE_TYPE]):
                self._error_cache.add(
                    InvalidTypeError(fully_qualified_name, spec, ATTRIBUTE_TYPE,
                                     InvalidTypeError.Reason.TYPE_NOT_DEFINED))

        self._spec_cache[fully_qualified_name] = spec
        for key, val in spec.items():
            if isinstance(val, list):
                for item in val:
                    self.add_schema_spec(item, fully_qualified_name)
            self.add_schema_spec(val, fully_qualified_name)

        return spec[ATTRIBUTE_NAME]