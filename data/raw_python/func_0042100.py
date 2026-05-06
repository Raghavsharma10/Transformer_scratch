def delete(self, *args, **kwargs):
        """
        Deletes the actual file from storage after the object is deleted.

        Calls super to actually delete the object.
        """
        file_obj = self.file
        super(AssetBase, self).delete(*args, **kwargs)
        self.delete_real_file(file_obj)