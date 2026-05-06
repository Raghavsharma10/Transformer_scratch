def spectrogram(self, x, n_fft=256, step=None, fs=1., window='hann',
                    color_scale='log', cmap='cubehelix', clim='auto'):
        """Calculate and show a spectrogram

        Parameters
        ----------
        x : array-like
            1D signal to operate on. ``If len(x) < n_fft``, x will be
            zero-padded to length ``n_fft``.
        n_fft : int
            Number of FFT points. Much faster for powers of two.
        step : int | None
            Step size between calculations. If None, ``n_fft // 2``
            will be used.
        fs : float
            The sample rate of the data.
        window : str | None
            Window function to use. Can be ``'hann'`` for Hann window, or None
            for no windowing.
        color_scale : {'linear', 'log'}
            Scale to apply to the result of the STFT.
            ``'log'`` will use ``10 * log10(power)``.
        cmap : str
            Colormap name.
        clim : str | tuple
            Colormap limits. Should be ``'auto'`` or a two-element tuple of
            min and max values.

        Returns
        -------
        spec : instance of Spectrogram
            The spectrogram.

        See also
        --------
        Image
        """
        self._configure_2d()
        # XXX once we have axes, we should use "fft_freqs", too
        spec = scene.Spectrogram(x, n_fft, step, fs, window,
                                 color_scale, cmap, clim)
        self.view.add(spec)
        self.view.camera.set_range()
        return spec