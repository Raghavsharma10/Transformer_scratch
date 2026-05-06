def show(
        self,
        filename: Optional[str] = None,
        show_link: bool = True,
        auto_open: bool = True,
        detect_notebook: bool = True,
    ) -> None:
        """Display the chart.

        Parameters
        ----------
        filename : str, optional
            Save plot to this filename, otherwise it's saved to a temporary file.
        show_link : bool, optional
            Show link to plotly.
        auto_open : bool, optional
            Automatically open the plot (in the browser).
        detect_notebook : bool, optional
            Try to detect if we're running in a notebook.

        """
        kargs = {}
        if detect_notebook and _detect_notebook():
            py.init_notebook_mode()
            plot = py.iplot
        else:
            plot = py.plot
            if filename is None:
                filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
            kargs['filename'] = filename
            kargs['auto_open'] = auto_open

        plot(self, show_link=show_link, **kargs)