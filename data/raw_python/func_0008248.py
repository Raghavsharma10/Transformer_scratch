def unfreeze(self):
        """
        Reverse of freeze
        """
        self.app.enable()
        self.clear.enable()
        self.nod.enable()
        self.led.enable()
        self.dummy.enable()
        self.readSpeed.enable()
        self.expose.enable()
        self.number.enable()
        self.wframe.enable()
        self.nmult.enable()
        self.frozen = False