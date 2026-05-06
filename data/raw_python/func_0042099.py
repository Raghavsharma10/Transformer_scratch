def save(self, *args, **kwargs):
        """
        For new assets, creates a new slug.
        For updates, deletes the old file from storage.

        Calls super to actually save the object.
        """
        if not self.pk and not self.slug:
            self.slug = self.generate_slug()

        if self.__original_file and self.file != self.__original_file:
            self.delete_real_file(self.__original_file)

        file_changed = True
        if self.pk:
            new_value = getattr(self, 'file')
            if hasattr(new_value, "file"):
                file_changed = isinstance(new_value.file, UploadedFile)
        else:
            self.cbversion = 0

        if file_changed:
            self.user_filename = os.path.basename(self.file.name)
            self.cbversion = self.cbversion + 1

        if not self.title:
            self.title = self.user_filename

        super(AssetBase, self).save(*args, **kwargs)

        if file_changed:
            signals.file_saved.send(self.file.name)
            utils.update_cache_bust_version(self.file.url, self.cbversion)
            self.reset_crops()

        if self.__original_file and self.file.name != self.__original_file.name:
            with manager.SwitchSchemaManager(None):
                for related in self.__class__._meta.get_all_related_objects(
                        include_hidden=True):
                    field = related.field
                    if getattr(field, 'denormalize', None):
                        cname = field.get_denormalized_field_name(field.name)
                        if getattr(field, 'denormalize'):
                            related.model.objects.filter(**{
                                field.name: self.pk
                            }).update(**{
                                cname: self.file.name
                            })