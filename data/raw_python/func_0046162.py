def make_pushable(collector, **kwargs):
    """Make only the pushable images and their dependencies"""
    configuration = collector.configuration
    configuration["harpoon"].do_push = True
    configuration["harpoon"].only_pushable = True
    make_all(collector, **kwargs)