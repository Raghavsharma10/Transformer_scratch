def show(collector, **kwargs):
    """Show what images we have"""
    configuration = collector.configuration
    flat = configuration.get("harpoon.flat", False)
    only_pushable = configuration.get("harpoon.only_pushable", False)

    for index, layer in enumerate(Builder().layered(configuration["images"], only_pushable=only_pushable)):
        if flat:
            for _, image in layer:
                print(image.image_name)
        else:
            print("Layer {0}".format(index))
            for _, image in layer:
                print("    {0}".format(image.display_line()))
            print("")