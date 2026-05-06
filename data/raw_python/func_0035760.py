def _document_structure(self):
        """Document the structure of the dataset."""
        logger.debug("Documenting dataset structure")
        key = self.get_structure_key()
        text = json.dumps(self._structure_parameters, indent=2, sort_keys=True)
        self.put_text(key, text)

        key = self.get_dtool_readme_key()
        self.put_text(key, self._dtool_readme_txt)