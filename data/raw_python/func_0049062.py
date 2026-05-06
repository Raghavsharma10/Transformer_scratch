def _is_in_set(self, inpt, metadata):
        """checks if the input is in the metadata's *_set list"""
        # makes an assumption there is only one _set in the metadata dict
        get_set_methods = [m for m in dir(metadata) if 'get_' in m and '_set' in m]
        set_results = None
        for m in get_set_methods:
            try:
                set_results = getattr(metadata, m)()
                break
            except errors.IllegalState:
                pass
        if set_results is not None and inpt in set_results:
            return True
        return False