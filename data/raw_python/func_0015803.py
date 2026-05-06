def add_rollback_panels(self):
        """
        Adds rollback panel to applicable model class's edit handlers.
        """
        from wagtailplus.utils.edit_handlers import add_panel_to_edit_handler
        from wagtailplus.wagtailrollbacks.edit_handlers import HistoryPanel

        for model in self.applicable_models:
            add_panel_to_edit_handler(model, HistoryPanel, _(u'History'))