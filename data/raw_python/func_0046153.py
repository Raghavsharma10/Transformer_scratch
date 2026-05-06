def push_all(collector, **kwargs):
    """Push all the images"""
    configuration = collector.configuration
    configuration["harpoon"].do_push = True
    configuration["harpoon"].only_pushable = True
    make_all(collector, **kwargs)