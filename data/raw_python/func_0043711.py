def send_zone_event(self, zone_id, event_name, *args):
        """ Send an event to a zone. """
        cmd = "EVENT %s!%s %s" % (
                zone_id.device_str(), event_name,
                " ".join(str(x) for x in args))
        return (yield from self._send_cmd(cmd))