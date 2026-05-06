def respond(self, obj):
        """Gives a response JSON(P) message"""

        # Get the callback argument if JSONP is allowed
        callback = self.get_argument("callback", None) if oz.settings["allow_jsonp"] else None

        # We're pretty strict with what callback names are allowed, just in case
        if callback and not CALLBACK_VALIDATOR.match(callback):
            raise oz.json_api.ApiError("Invalid callback identifier - only functions with ASCII characters are allowed")

        # Provide the response in a different manner depending on whether a
        # JSONP callback is specified
        json = escape.json_encode(obj)

        if callback:
            self.set_header("Content-Type", "application/javascript; charset=UTF-8")
            self.finish("%s(%s)" % (callback, json))
        else:
            self.set_header("Content-Type", "application/json; charset=UTF-8")
            self.finish(json)