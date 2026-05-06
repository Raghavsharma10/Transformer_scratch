def __sweeten(self, dumper: 'Dumper', class_: Type, node: Node) -> None:
        """Applies the user's yatiml_sweeten() function(s), if any.

        Sweetening is done for the base classes first, then for the \
        derived classes, down the hierarchy to the class we're \
        constructing.

        Args:
            dumper: The dumper that is dumping this object.
            class_: The type of the object to be dumped.
            represented_object: The object to be dumped.
        """
        for base_class in class_.__bases__:
            if base_class in dumper.yaml_representers:
                logger.debug('Sweetening for class {}'.format(
                    self.class_.__name__))
                self.__sweeten(dumper, base_class, node)
        if hasattr(class_, 'yatiml_sweeten'):
            class_.yatiml_sweeten(node)