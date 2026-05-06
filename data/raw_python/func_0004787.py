def diagnostics_contpix(self, data, nchunks=10, fig = "baseline_spec_with_cont_pix"):
        """ Call plot_contpix once for each nth of the spectrum """
        if data.contmask is None:
            print("No contmask set")
        else:
            coeffs_all = self.coeffs
            wl = data.wl
            baseline_spec = coeffs_all[:,0]
            contmask = data.contmask
            contpix_x = wl[contmask]
            contpix_y = baseline_spec[contmask]
            rem = len(wl)%nchunks
            wl_split = np.array(np.split(wl[0:len(wl)-rem],nchunks))
            baseline_spec_split = np.array(
                    np.split(baseline_spec[0:len(wl)-rem],nchunks))
            nchunks = wl_split.shape[0]
            for i in range(nchunks):
                fig_chunk = fig + "_%s" %str(i)
                wl_chunk = wl_split[i,:]
                baseline_spec_chunk = baseline_spec_split[i,:]
                take = np.logical_and(
                        contpix_x>wl_chunk[0], contpix_x<wl_chunk[-1])
                self.plot_contpix(
                        wl_chunk, baseline_spec_chunk, 
                        contpix_x[take], contpix_y[take], fig_chunk)