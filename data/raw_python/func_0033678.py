def _on_light_state(self, header, payload, rest, addr):
        """
        Records the light state of bulbs, and forwards to a high-level callback
        with human-friendlier arguments.
        """
        with self.lock:
            label = payload['label'].strip('\x00')
            self.bulbs[header.mac] = bulb = Bulb(label, header.mac)
            if len(self.bulbs) >= self.num_bulbs:
                self.bulbs_found_event.set()

            self.light_state[header.mac] = payload
            if len(self.light_state) >= self.num_bulbs:
                self.light_state_event.set()

        self.callbacks.put(EVENT_LIGHT_STATE, bulb,
                           raw=payload,
                           hue=(payload['hue'] / float(0xffff) * 360) % 360.0,
                           saturation=payload['sat'] / float(0xffff),
                           brightness=payload['bright'] / float(0xffff),
                           kelvin=payload['kelvin'],
                           is_on=bool(payload['power']))