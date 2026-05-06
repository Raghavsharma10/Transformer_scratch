def script_start_type(script):
        """Return the type of block the script begins with."""
        if script[0].type.text == 'when @greenFlag clicked':
            return HairballPlugin.HAT_GREEN_FLAG
        elif script[0].type.text == 'when I receive %s':
            return HairballPlugin.HAT_WHEN_I_RECEIVE
        elif script[0].type.text == 'when this sprite clicked':
            return HairballPlugin.HAT_MOUSE
        elif script[0].type.text == 'when %s key pressed':
            return HairballPlugin.HAT_KEY
        else:
            return HairballPlugin.NO_HAT