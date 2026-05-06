def clean_bucket_name(self):
        """Validates that the bucket name in the provided data matches the
        bucket name from the storage backend."""
        bucket_name = self.cleaned_data['bucket_name']
        if not bucket_name == self.get_bucket_name():
            raise forms.ValidationError('Bucket name does not validate.')
        return bucket_name