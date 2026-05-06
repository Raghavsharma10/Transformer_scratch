def _add_backend(self, backend):
        """ Builds a subclass model for the given backend """
        md_type = backend.verbose_name
        base = backend().get_model(self)
        # TODO: Rename this field
        new_md_attrs = {'_metadata': self.metadata, '__module__': __name__ }

        new_md_meta = {}
        new_md_meta['verbose_name'] = '%s (%s)' % (self.verbose_name, md_type)
        new_md_meta['verbose_name_plural'] = '%s (%s)' % (self.verbose_name_plural, md_type)
        new_md_meta['unique_together'] = base._meta.unique_together
        new_md_attrs['Meta'] = type("Meta", (), new_md_meta)
        new_md_attrs['_metadata_type'] = backend.name
        model = type("%s%s"%(self.name,"".join(md_type.split())), (base, self.MetadataBaseModel), new_md_attrs.copy())
        self.models[backend.name] = model
        # This is a little dangerous, but because we set __module__ to __name__, the model needs tobe accessible here
        globals()[model.__name__] = model