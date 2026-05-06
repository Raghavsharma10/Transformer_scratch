def getRtplotWins(self):
        """"
        Returns a string suitable to sending off to rtplot when
        it asks for window parameters. Returns null string '' if
        the windows are not OK. This operates on the basis of
        trying to send something back, even if it might not be
        OK as a window setup. Note that we have to take care
        here not to update any GUI components because this is
        called outside of the main thread.
        """
        try:
            if self.isFF():
                return 'fullframe\r\n'
            elif self.isDrift():
                xbin = self.wframe.xbin.value()
                ybin = self.wframe.ybin.value()
                nwin = 2*self.wframe.npair.value()
                ret = str(xbin) + ' ' + str(ybin) + ' ' + str(nwin) + '\r\n'
                for xsl, xsr, ys, nx, ny in self.wframe:
                    ret += '{:d} {:d} {:d} {:d}\r\n'.format(
                        xsl, ys, nx, ny
                    )
                    ret += '{:d} {:d} {:d} {:d}'.format(
                        xsr, ys, nx, ny
                    )
                return ret
            else:
                xbin = self.wframe.xbin.value()
                ybin = self.wframe.ybin.value()
                nwin = 4*self.wframe.nquad.value()
                ret = str(xbin) + ' ' + str(ybin) + ' ' + str(nwin) + '\r\n'
                for xsll, xsul, xslr, xsur, ys, nx, ny in self.wframe:
                    ret += '{:d} {:d} {:d} {:d}\r\n'.format(
                        xsll, ys, nx, ny
                    )
                    ret += '{:d} {:d} {:d} {:d}\r\n'.format(
                        xsul, 1025 - ys - ny, nx, ny
                    )
                    ret += '{:d} {:d} {:d} {:d}\r\n'.format(
                        xslr, ys, nx, ny
                    )
                    ret += '{:d} {:d} {:d} {:d}\r\n'.format(
                        xsur, 1025 - ys - ny, nx, ny
                    )
                return ret
        except:
            return ''