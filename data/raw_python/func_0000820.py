def wait_for_window_map_state(self, window, state):
        """
        Wait for a window to have a specific map state.

        State possibilities:
          IsUnmapped - window is not displayed.
          IsViewable - window is mapped and shown (though may be
              clipped by windows on top of it)
          IsUnviewable - window is mapped but a parent window is unmapped.

        :param window: the window you want to wait for.
        :param state: the state to wait for.
        """
        _libxdo.xdo_wait_for_window_map_state(self._xdo, window, state)