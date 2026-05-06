def save(
        self,
        filename: Optional[str] = None,
        show_link: bool = True,
        auto_open: bool = False,
        output: str = 'file',
        plotlyjs: bool = True,
    ) -> str:
        """Save the chart to an html file."""
        if filename is None:
            filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
        # NOTE: this doesn't work for output 'div'
        py.plot(
            self,
            show_link=show_link,
            filename=filename,
            auto_open=auto_open,
            output_type=output,
            include_plotlyjs=plotlyjs,
        )
        return filename