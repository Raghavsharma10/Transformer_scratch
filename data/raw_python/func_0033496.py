def copy(self, new_path, replace=False):
        """Uses boto to copy the file to the new path instead of uploading another file to the new key"""
        if replace or not get_file(new_path).exists():
            self.key.copy(self.key.bucket, new_path)
            return True
        return False