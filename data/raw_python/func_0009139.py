def serialize_dict(self, msg_dict):
        """Serialize to JSON a message dictionary."""
        serialized = json.dumps(msg_dict, namedtuple_as_object=False)
        if PY2:
            serialized = serialized.decode("utf-8")
        serialized = "{}\nend\n".format(serialized)
        return serialized