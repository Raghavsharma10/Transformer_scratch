def _timer(self, state_transition_event=None):
        """Timer loop used to keep track of the time while roasting or
        cooling. If the time remaining reaches zero, the roaster will call the
        supplied state transistion function or the roaster will be set to
        the idle state."""
        while not self._teardown.value:
            state = self.get_roaster_state()
            if(state == 'roasting' or state == 'cooling'):
                time.sleep(1)
                self.total_time += 1
                if(self.time_remaining > 0):
                    self.time_remaining -= 1
                else:
                    if(state_transition_event is not None):
                        state_transition_event.set()
                    else:
                        self.idle()
            else:
                time.sleep(0.01)