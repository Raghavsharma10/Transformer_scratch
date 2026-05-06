def _api_on_write_error(self, status_code, **kwargs):
        """
        Catches errors and renders it as a JSON message. Adds the traceback if
        debug is enabled.
        """

        return_error = { "code": self.get_status() }
        exc_info = kwargs.get("exc_info")

        if exc_info and isinstance(exc_info[1], oz.json_api.ApiError):
            return_error["error"] = exc_info[1].message
        else:
            return_error["error"] = API_ERROR_CODE_MAP.get(self.get_status(), "Unknown error")

        if oz.settings.get("debug"):
            return_error["trace"] = "".join(traceback.format_exception(*exc_info))

        self.finish(return_error)
        return oz.break_trigger