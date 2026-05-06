def _init(self, style, streamer, processors=None):
        """Do writer-specific setup.

        Parameters
        ----------
        style : dict
            Style, as passed to __init__.
        streamer : interface.Stream
            A stream interface that takes __init__'s `stream` and `interactive`
            arguments into account.
        processors : field.StyleProcessors, optional
            A writer-specific processors instance.  Defaults to
            field.PlainProcessors().
        """
        self._stream = streamer
        if streamer.interactive:
            if streamer.supports_updates:
                self.mode = "update"
            else:
                self.mode = "incremental"
        else:
            self.mode = "final"

        if style and "width_" not in style and self._stream.width:
            style["width_"] = self._stream.width
        self._content = ContentWithSummary(
            StyleFields(style, processors or PlainProcessors()))