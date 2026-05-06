def create(cls, data):
        """ Create a new event instance.

        Return an instance of the `GerritEvent` subclass after converting
        `data` to json.

        Raise GerritError if json parsed from `data` does not contain a `type`
        key.

        """
        try:
            json_data = json.loads(data)
        except ValueError as err:
            logging.debug("Failed to load json data: %s: [%s]", str(err), data)
            json_data = json.loads(ErrorEvent.error_json(err))

        if "type" not in json_data:
            raise GerritError("`type` not in json_data")
        name = json_data["type"]
        if name not in cls._events:
            name = 'unhandled-event'
        event = cls._events[name]
        module_name = event[0]
        class_name = event[1]
        module = __import__(module_name, fromlist=[module_name])
        klazz = getattr(module, class_name)
        return klazz(json_data)