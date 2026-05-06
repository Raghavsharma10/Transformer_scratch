def __update_display_items_model(self, display_items_model: ListModel.FilteredListModel, data_group: typing.Optional[DataGroup.DataGroup], filter_id: typing.Optional[str]) -> None:
        """Update the data item model with a new container, filter, and sorting.

        This is called when the data item model is created or when the user changes
        the data group or sorting settings.
        """

        with display_items_model.changes():  # change filter and sort together
            if data_group is not None:
                display_items_model.container = data_group
                display_items_model.filter = ListModel.Filter(True)
                display_items_model.sort_key = None
                display_items_model.filter_id = None
            elif filter_id == "latest-session":
                display_items_model.container = self.document_model
                display_items_model.filter = ListModel.EqFilter("session_id", self.document_model.session_id)
                display_items_model.sort_key = DataItem.sort_by_date_key
                display_items_model.sort_reverse = True
                display_items_model.filter_id = filter_id
            elif filter_id == "temporary":
                display_items_model.container = self.document_model
                display_items_model.filter = ListModel.NotEqFilter("category", "persistent")
                display_items_model.sort_key = DataItem.sort_by_date_key
                display_items_model.sort_reverse = True
                display_items_model.filter_id = filter_id
            elif filter_id == "none":  # not intended to be used directly
                display_items_model.container = self.document_model
                display_items_model.filter = ListModel.Filter(False)
                display_items_model.sort_key = DataItem.sort_by_date_key
                display_items_model.sort_reverse = True
                display_items_model.filter_id = filter_id
            else:  # "all"
                display_items_model.container = self.document_model
                display_items_model.filter = ListModel.EqFilter("category", "persistent")
                display_items_model.sort_key = DataItem.sort_by_date_key
                display_items_model.sort_reverse = True
                display_items_model.filter_id = None