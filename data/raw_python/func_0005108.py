def loadinfofont(self):
        """Auxiliary method to load font if not yet done."""
        if self.infofont == None:
            self.infofont = imft.load_path(os.path.join(fontsdir, "courR10.pil"))