def clean_descriptor(self, number):
        """Close file descriptor and remove underlying files."""
        self.descriptors[number]['stdout'].close()
        self.descriptors[number]['stderr'].close()

        if os.path.exists(self.descriptors[number]['stdout_path']):
            os.remove(self.descriptors[number]['stdout_path'])

        if os.path.exists(self.descriptors[number]['stderr_path']):
            os.remove(self.descriptors[number]['stderr_path'])