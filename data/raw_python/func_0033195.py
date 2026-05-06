def ModelDir(self):
        """Absolute FilePath to the training output directory.
        """
        model_dir = self.Parameters['model_output_dir'].Value
        absolute_model_dir = os.path.abspath(model_dir)
        return FilePath(absolute_model_dir)