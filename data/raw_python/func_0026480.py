def _page_update(self, event):
        """
        Checks if the newly created object is a wikipage..
        If so, rerenders the automatic index.

        :param event: objectchange or objectcreation event
        """
        try:
            if event.schema == 'wikipage':
                self._update_index()

        except Exception as e:
            self.log("Page creation notification error: ", event, e,
                     type(e), lvl=error)