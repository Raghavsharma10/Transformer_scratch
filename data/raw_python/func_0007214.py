def _find_class(self, class_name):
        "Resolve the class from the name."
        classes = {}
        classes.update(globals())
        classes.update(self.INSTANCE_CLASSES)
        logger.debug(f'looking up class: {class_name}')
        cls = classes[class_name]
        logger.debug(f'found class: {cls}')
        return cls