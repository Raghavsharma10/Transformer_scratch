def clean_key_name(self):
        """Validates that the key in the provided data starts with the
        required prefix, and that it exists in the bucket."""
        key = self.cleaned_data['key_name']
        # Ensure key starts with prefix
        if not key.startswith(self.get_key_prefix()):
            raise forms.ValidationError('Key does not have required prefix.')
        # Ensure key exists
        if not self.get_upload_key():
            raise forms.ValidationError('Key does not exist.')
        return key