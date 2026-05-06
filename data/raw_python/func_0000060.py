def attach(*layouts, **kwargs):
    """
    Registers the given layout(s) classes
    admin site:

    @pages.register(Page)
    class Default(PageLayout):
        pass
    """

    def _model_admin_wrapper(layout_class):
        register(layout_class, layouts[0])
        return layout_class
    return _model_admin_wrapper