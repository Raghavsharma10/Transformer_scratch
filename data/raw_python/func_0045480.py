def add_to_dumper(dumper: Type, classes: List[Type]) -> None:
    """Register user-defined classes with the Dumper.

    This enables the Dumper to write objects of your classes to a \
    YAML file. Note that all the arguments are types, not instances!

    Args:
        dumper: Your dumper class(!), derived from yatiml.Dumper
        classes: One or more classes to add.
    """
    if not isinstance(classes, list):
        classes = [classes]  # type: ignore
    for class_ in classes:
        if issubclass(class_, enum.Enum):
            dumper.add_representer(class_, EnumRepresenter(class_))
        elif issubclass(class_, str) or issubclass(class_, UserString):
            dumper.add_representer(class_, UserStringRepresenter(class_))
        else:
            dumper.add_representer(class_, Representer(class_))