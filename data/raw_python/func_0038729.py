def render_summary(self, include_title=True):
        """Render the traceback for the interactive console."""
        title = ''
        description = ''
        frames = []
        classes = ['traceback']
        if not self.frames:
            classes.append('noframe-traceback')

        if include_title:
            if self.is_syntax_error:
                title = text_('Syntax Error')
            else:
                title = text_('Traceback <em>(most recent call last)</em>:')

        for frame in self.frames:
            frames.append(text_('<li%s>%s') % (
                frame.info and text_(' title="%s"') % escape(frame.info) or text_(''),
                frame.render()
                ))

        if self.is_syntax_error:
            description_wrapper = text_('<pre class=syntaxerror>%s</pre>')
        else:
            description_wrapper = text_('<blockquote>%s</blockquote>')

        return SUMMARY_HTML % {
            'classes':      text_(' '.join(classes)),
            'title':        title and text_('<h3>%s</h3>' % title) or text_(''),
            'frames':       text_('\n'.join(frames)),
            'description':  description_wrapper % escape(self.exception)
        }