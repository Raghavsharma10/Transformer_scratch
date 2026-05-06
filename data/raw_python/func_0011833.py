def wait_for_keys(self, *keys, timeout=0):
        """Waits until one of the specified keys was pressed, and returns 
        which key was pressed.

        :param keys: iterable of integers of pygame-keycodes, or simply 
            multiple keys passed via multiple arguments
        :type keys: iterable
        :param timeout: number of seconds to wait till the function returns
        :type timeout: float

        :returns: The keycode of the pressed key, or None in case of timeout
        :rtype: int
        """
        if len(keys) == 1 and _is_iterable(keys[0]):
            keys = keys[0]

        return self.listen_until_return(Handler.key_press(keys), timeout=timeout)