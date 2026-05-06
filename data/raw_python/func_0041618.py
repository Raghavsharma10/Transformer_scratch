def compile(self, source_code, post_treatment=''.join):
        """Compile given source code.
        Return object code, modified by given post treatment.
        """
        # read structure
        structure = self._structure(source_code)
        values    = self._struct_to_values(structure, source_code)
        # create object code, translated in targeted language
        obj_code = langspec.translated(
            structure, values, 
            self.target_lang_spec
        )
        # apply post treatment and return
        return obj_code if post_treatment is None else post_treatment(obj_code)