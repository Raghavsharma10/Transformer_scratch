def json_loads(payload):
        "Log the payload that cannot be parsed"
        try:
            return json.loads(payload)
        except ValueError as e:
            log.error("unable to json.loads(" + payload + ")")
            raise e