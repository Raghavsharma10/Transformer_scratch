def _prepare_output_multi(self, model):
        """If printing to a different file per model, change the file for the current model"""
        model_name = model.__name__
        current_path = os.path.join(self._output_path, '{model}.{extension}'.format(
            model=model_name,
            extension=self.EXTENSION,
        ))
        self._outfile = codecs.open(current_path, 'w', encoding='utf-8')
        print('Dumping {model} to {file}'.format(model=model_name, file=current_path))