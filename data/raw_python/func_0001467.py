def get_qpimage(self, index):
        """Return a single QPImage of the series

        Parameters
        ----------
        index: int or str
            Index or identifier of the QPImage

        Notes
        -----
        Instead of ``qps.get_qpimage(index)``, it is possible
        to use the short-hand ``qps[index]``.
        """
        if isinstance(index, str):
            # search for the identifier
            for ii in range(len(self)):
                qpi = self[ii]
                if "identifier" in qpi and qpi["identifier"] == index:
                    group = self.h5["qpi_{}".format(ii)]
                    break
            else:
                msg = "QPImage identifier '{}' not found!".format(index)
                raise KeyError(msg)
        else:
            # integer index
            if index < -len(self):
                msg = "Index {} out of bounds for QPSeries of size {}!".format(
                    index, len(self))
                raise ValueError(msg)
            elif index < 0:
                index += len(self)
            name = "qpi_{}".format(index)
            if name in self.h5:
                group = self.h5[name]
            else:
                msg = "Index {} not found for QPSeries of length {}".format(
                    index, len(self))
                raise KeyError(msg)
        return QPImage(h5file=group)