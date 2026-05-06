def write(self) -> None:
        """Call method |NetCDFFile.write| of all handled |NetCDFFile| objects.
        """
        if self.folders:
            init = hydpy.pub.timegrids.init
            timeunits = init.firstdate.to_cfunits('hours')
            timepoints = init.to_timepoints('hours')
            for folder in self.folders.values():
                for file_ in folder.values():
                    file_.write(timeunits, timepoints)