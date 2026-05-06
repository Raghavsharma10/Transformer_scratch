def key_press(keys):
        """returns a handler that can be used with EventListener.listen()
        and returns when a key in keys is pressed"""
        return lambda e: e.key if e.type == pygame.KEYDOWN \
            and e.key in keys else EventConsumerInfo.DONT_CARE