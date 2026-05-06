def run1(self):
    """Run one item (a callback or an RPC wait_any) or sleep.

    Returns:
      True if something happened; False if all queues are empty.
    """
    delay = self.run0()
    if delay is None:
      return False
    if delay > 0:
      self.clock.sleep(delay)
    return True