def _handle_changed_fields(self, old_data):
        """
        Looks for changed relation fields between new and old data (before/after save).
        Creates back_link references for updated fields.

        Args:
            old_data: Object's data before save.
        """
        for link in self.get_links(is_set=False):
            fld_id = un_camel_id(link['field'])
            if not old_data or old_data.get(fld_id) != self._data[fld_id]:
                # self is new or linked model changed
                if self._data[fld_id]:  # exists
                    linked_mdl = getattr(self, link['field'])
                    self._add_back_link(linked_mdl, link)