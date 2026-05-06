def parse_view_args(self, req, name, field):
        """Pull a value from the request's ``view_args``."""
        return core.get_value(req.match_info, name, field)