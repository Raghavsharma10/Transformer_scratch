def _view_filter(self):
        """
        Overrides OsidSession._view_filter to add sequestering filter.

        """
        view_filter = OsidSession._view_filter(self)
        if self._sequestered_view == SEQUESTERED:
            view_filter['sequestered'] = False
        return view_filter