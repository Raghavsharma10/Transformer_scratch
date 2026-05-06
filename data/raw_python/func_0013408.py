def _init_optional_attrs(self, optional_attrs):
        """Prepare to store data from user-desired optional fields.

          Not loading these optional fields by default saves in space and speed.
          But allow the possibility for saving these fields, if the user desires,
            Including:
              comment consider def is_class_level is_metadata_tag is_transitive
              relationship replaced_by subset synonym transitive_over xref
        """
        # Written by DV Klopfenstein
        # Required attributes are always loaded. All others are optionally loaded.
        self.attrs_req = ['id', 'alt_id', 'name', 'namespace', 'is_a', 'is_obsolete']
        self.attrs_scalar = ['comment', 'defn',
                             'is_class_level', 'is_metadata_tag',
                             'is_transitive', 'transitive_over']
        self.attrs_nested = frozenset(['relationship'])
        # Allow user to specify either: 'def' or 'defn'
        #   'def' is an obo field name, but 'defn' is legal Python attribute name
        fnc = lambda aopt: aopt if aopt != "defn" else "def"
        if optional_attrs is None:
            optional_attrs = []
        elif isinstance(optional_attrs, str):
            optional_attrs = [fnc(optional_attrs)] if optional_attrs not in self.attrs_req else []
        elif isinstance(optional_attrs, list) or isinstance(optional_attrs, set):
            optional_attrs = set([fnc(f) for f in optional_attrs if f not in self.attrs_req])
        else:
            raise Exception("optional_attrs arg MUST BE A str, list, or set.")
        self.optional_attrs = optional_attrs