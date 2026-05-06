def list_catalogs(self):
        """
        Lists existing catalogs respect to ui view template format
        """
        _form = CatalogSelectForm(current=self.current)
        _form.set_choices_of('catalog', [(i, i) for i in fixture_bucket.get_keys()])
        self.form_out(_form)