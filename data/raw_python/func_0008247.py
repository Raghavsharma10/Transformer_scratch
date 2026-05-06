def freeze(self):
        """
        Freeze all settings so they cannot be altered
        """
        self.app.disable()
        self.clear.disable()
        self.nod.disable()
        self.led.disable()
        self.dummy.disable()
        self.readSpeed.disable()
        self.expose.disable()
        self.number.disable()
        self.wframe.disable(everything=True)
        self.nmult.disable()
        self.frozen = True