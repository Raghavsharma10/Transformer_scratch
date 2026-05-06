def print_all_dockerfiles(collector, **kwargs):
    """Print all the dockerfiles"""
    for name, image in collector.configuration["images"].items():
        print("{0}".format(name))
        print("-" * len(name))
        kwargs["image"] = image
        print_dockerfile(collector, **kwargs)