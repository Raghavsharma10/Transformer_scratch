def control(self, on=[], off=[]):
        """
        This method serves as the primary interaction point
            to the controls interface.
        - The 'on' and 'off' arguments can either be a list or a single string.
            This allows for both individual device control and batch controls.

        Note:
            Both the onlist and offlist are optional. 
            If only one item is being managed, it can be passed as a string.

        Usage:
            - Turning off all devices:
                ctrlobj.control(off="all")
            - Turning on all devices:
                ctrlobj.control(on="all")

            - Turning on the light and fan ONLY (for example)
                ctrlobj.control(on=["light", "fan"])

            - Turning on the light and turning off the fan (for example)
                ctrolobj.control(on="light", off="fan")

        """
        controls = {"light", "valve", "fan", "pump"}

        def cast_arg(arg):
            if type(arg) is str:
                if arg == "all":
                    return controls
                else:
                    return {arg} & controls
            else:
                return set(arg) & controls

        # User has requested individual controls.
        for item in cast_arg(on):
            self.manage(item, "on")
        for item in cast_arg(off):
            self.manage(item, "off")
        sleep(.01) # Force delay to throttle requests
        return self.update()