def _class_name_params(self, name):
        "Get the class name and parameters to use for ``__init__``."
        sec = self.pattern.format(**{'name': name})
        logger.debug(f'section: {sec}')
        params = {}
        params.update(self.config.populate({}, section=sec))
        class_name = params['class_name']
        del params['class_name']
        return class_name, params