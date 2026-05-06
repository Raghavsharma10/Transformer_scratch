def _plot_debug_slopes_directions(self):
        """
        A debug function to plot the direction calculated in various ways.
        """
        # %%
        from matplotlib.pyplot import matshow, colorbar, clim, title

        matshow(self.direction / np.pi * 180); colorbar(); clim(0, 360)
        title('Direction')

        mag2, direction2 = self._central_slopes_directions()
        matshow(direction2 / np.pi * 180.0); colorbar(); clim(0, 360)
        title('Direction (central difference)')

        matshow(self.mag); colorbar()
        title('Magnitude')
        matshow(mag2); colorbar(); title("Magnitude (Central difference)")

        # %%
        # Compare to Taudem
        filename = self.file_name
        os.chdir('testtiff')
        try:
            os.remove('test_ang.tif')
            os.remove('test_slp.tif')
        except:
            pass
        cmd = ('dinfflowdir -fel "%s" -ang "%s" -slp "%s"' %
               (os.path.split(filename)[-1], 'test_ang.tif', 'test_slp.tif'))
        taudem._run(cmd)

        td_file = GdalReader(file_name='test_ang.tif')
        td_ang, = td_file.raster_layers
        td_file2 = GdalReader(file_name='test_slp.tif')
        td_mag, = td_file2.raster_layers
        os.chdir('..')

        matshow(td_ang.raster_data / np.pi*180); clim(0, 360); colorbar()
        title('Taudem direction')
        matshow(td_mag.raster_data); colorbar()
        title('Taudem magnitude')

        matshow(self.data); colorbar()
        title('The test data (elevation)')

        diff = (td_ang.raster_data - self.direction) / np.pi * 180.0
        diff[np.abs(diff) > 300] = np.nan
        matshow(diff); colorbar(); clim([-1, 1])
        title('Taudem direction - calculated Direction')

        # normalize magnitudes
        mag2 = td_mag.raster_data
        mag2 /= np.nanmax(mag2)
        mag = self.mag.copy()
        mag /= np.nanmax(mag)
        matshow(mag - mag2); colorbar()
        title('Taudem magnitude - calculated magnitude')
        del td_file
        del td_file2
        del td_ang
        del td_mag