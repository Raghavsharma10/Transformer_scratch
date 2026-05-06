def _on_power_state(self, header, payload, rest, addr):
        """
        Records the power (on/off) state of bulbs, and forwards to a high-level
        callback with human-friendlier arguments.
        """
        with self.lock:
            self.power_state[header.mac] = payload
            if len(self.power_state) >= self.num_bulbs:
                self.power_state_event.set()

        self.callbacks.put(EVENT_POWER_STATE, self.get_bulb(header.mac),
                           is_on=bool(payload['is_on']))