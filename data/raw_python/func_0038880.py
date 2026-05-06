def _json_struct(self):
        """The json data structure in the URL contents, it will cache this
           if it makes sense so it doesn't parse over and over."""
        if self.__has_json__:
            if self.__cache_request__:
                if self.__json_struct__ is Ellipsis:
                    if self._contents is not Ellipsis:
                        self.__json_struct__ = json.loads(
                                           compat.ensure_string(self._contents)
                                                              .strip() or '{}')
                    else:
                        return {}
                return self.__json_struct__
            else:
                return json.loads(compat.ensure_string(self._contents))
        else:
            # Return an empty dict for things so they don't have to special
            # case against a None value or anything
            return {}