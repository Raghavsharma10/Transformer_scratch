def get_catalog(self):
        """
        Get existing catalog and fill the form with the model data.
        If given key not found as catalog, it generates an empty catalog data form.
        """

        catalog_data = fixture_bucket.get(self.input['form']['catalog'])

        # define add or edit based on catalog data exists
        add_or_edit = "Edit" if catalog_data.exists else "Add"

        # generate form
        catalog_edit_form = CatalogEditForm(
            current=self.current,
            title='%s: %s' % (add_or_edit, self.input['form']['catalog']))

        # add model data to form
        if catalog_data.exists:
            if type(catalog_data.data) == list:
                # if catalog data is an array it means no other language of value defined, therefor the value is turkish
                for key, data in enumerate(catalog_data.data):
                    catalog_edit_form.CatalogDatas(catalog_key=key or "0", en='', tr=data)
            if type(catalog_data.data) == dict:
                for key, data in catalog_data.data.items():
                    catalog_edit_form.CatalogDatas(catalog_key=key, en=data['en'], tr=data['tr'])

        else:
            catalog_edit_form.CatalogDatas(catalog_key="0", en='', tr='')

        self.form_out(catalog_edit_form)

        # schema key for get back what key will be saved, used in save_catalog form
        self.output["object_key"] = self.input['form']['catalog']