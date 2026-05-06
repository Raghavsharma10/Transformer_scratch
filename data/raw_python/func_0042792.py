def new_instance(settings):
    """
    MAKE A PYTHON INSTANCE

    `settings` HAS ALL THE `kwargs`, PLUS `class` ATTRIBUTE TO INDICATE THE CLASS TO CREATE
    """
    settings = set_default({}, settings)
    if not settings["class"]:
        Log.error("Expecting 'class' attribute with fully qualified class name")

    # IMPORT MODULE FOR HANDLER
    path = settings["class"].split(".")
    class_name = path[-1]
    path = ".".join(path[:-1])
    constructor = None
    try:
        temp = __import__(path, globals(), locals(), [class_name], 0)
        constructor = object.__getattribute__(temp, class_name)
    except Exception as e:
        Log.error("Can not find class {{class}}", {"class": path}, cause=e)

    settings['class'] = None
    try:
        return constructor(kwargs=settings)  # MAYBE IT TAKES A KWARGS OBJECT
    except Exception as e:
        pass

    try:
        return constructor(**settings)
    except Exception as e:
        Log.error("Can not create instance of {{name}}", name=".".join(path), cause=e)