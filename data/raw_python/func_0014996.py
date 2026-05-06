def str2listtuple(self, string_message):
        "Covert a string that is ready to be sent to graphite into a tuple"

        if type(string_message).__name__ not in ('str', 'unicode'):
            raise TypeError("Must provide a string or unicode")

        if not string_message.endswith('\n'):
            string_message += "\n"

        tpl_list = []
        for line in string_message.split('\n'):
            line = line.strip()
            if not line:
                continue
            path, metric, timestamp = (None, None, None)
            try:
                (path, metric, timestamp) = line.split()
            except ValueError:
                raise ValueError(
                    "message must contain - metric_name, value and timestamp '%s'"
                    % line)
            try:
                timestamp = float(timestamp)
            except ValueError:
                raise ValueError("Timestamp must be float or int")

            tpl_list.append((path, (timestamp, metric)))

        if len(tpl_list) == 0:
            raise GraphiteSendException("No messages to send")

        payload = pickle.dumps(tpl_list)
        header = struct.pack("!L", len(payload))
        message = header + payload

        return message