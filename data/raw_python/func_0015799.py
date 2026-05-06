def add_relationship_panels(self):
        """
        Add edit handler that includes "related" panels to applicable
        model classes that don't explicitly define their own edit handler.
        """
        from wagtailplus.utils.edit_handlers import add_panel_to_edit_handler
        from wagtailplus.wagtailrelations.edit_handlers import RelatedPanel

        for model in self.applicable_models:
            add_panel_to_edit_handler(model, RelatedPanel, _(u'Related'))