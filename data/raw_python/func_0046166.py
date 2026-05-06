def show_pushable(collector, **kwargs):
    """Show what images we have"""
    collector.configuration['harpoon'].only_pushable = True
    show(collector, **kwargs)