def _pfp__handle_updated(self, watched_field):
        """Handle the watched field that was updated
        """
        self._pfp__no_notify = True

        # nested data has been changed, so rebuild the
        # nested data to update the field
        # TODO a global setting to determine this behavior?
        # could slow things down a bit for large nested structures

        # notice the use of _is_ here - 'is' != '=='. '==' uses
        # the __eq__ operator, while is compares id(object) results
        if watched_field is self._:
            self._pfp__pack_data()
        elif self._pfp__update_func is not None:
            self._pfp__update_func.call(
                [self] + self._pfp__watch_fields,
                *self._pfp__update_func_call_info
            )

        self._pfp__no_notify = False