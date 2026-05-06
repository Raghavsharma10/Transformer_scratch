def loadlabelfont(self):
        """Auxiliary method to load font if not yet done."""
        if self.labelfont == None:
            self.labelfont = imft.load_path(os.path.join(fontsdir, "courR10.pil"))