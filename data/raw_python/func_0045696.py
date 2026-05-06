def add_to_loader(loader_cls: Type, classes: List[Type]) -> None:
    """Registers one or more classes with a YAtiML loader.

    Once a class has been registered, it can be recognized and \
    constructed when reading a YAML text.

    Args:
        loader_cls: The loader to register the classes with.
        classes: The class(es) to register, a plain Python class or a \
                list of them.
    """
    if not isinstance(classes, list):
        classes = [classes]  # type: ignore

    for class_ in classes:
        tag = '!{}'.format(class_.__name__)
        if issubclass(class_, enum.Enum):
            loader_cls.add_constructor(tag, EnumConstructor(class_))
        elif issubclass(class_, str) or issubclass(class_, UserString):
            loader_cls.add_constructor(tag, UserStringConstructor(class_))
        else:
            loader_cls.add_constructor(tag, Constructor(class_))

        if not hasattr(loader_cls, '_registered_classes'):
            loader_cls._registered_classes = dict()
        loader_cls._registered_classes[tag] = class_