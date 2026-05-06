def get_output_file(self, in_file, instance, field, **kwargs):
        """Creates a temporary file. With regular `FileSystemStorage` it does not 
        need to be deleted, instaed file is safely moved over. With other cloud
        based storage it is a good idea to set `delete=True`."""
        return NamedTemporaryFile(mode='rb', suffix='_%s_%s%s' % (
            get_model_name(instance), field.name, self.get_ext()), delete=False)