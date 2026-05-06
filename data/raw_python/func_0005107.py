def loadtitlefont(self):
        """Auxiliary method to load font if not yet done."""
        if self.titlefont == None:
#             print 'the bloody fonts dir is????', fontsdir
#             print 'pero esto que hace??', os.path.join(fontsdir, "courR18.pil")
#             /home/vital/Workspace/pyResources/Scientific_Lib/f2n_fonts/f2n_fonts/courR18.pil
#             /home/vital/Workspace/pyResources/Scientific_Lib/f2n_fonts
            self.titlefont = imft.load_path(os.path.join(fontsdir, "courR18.pil"))