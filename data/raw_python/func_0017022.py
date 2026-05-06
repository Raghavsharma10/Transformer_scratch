def markdown(self, text, mode='', context='', raw=False):
        """Render an arbitrary markdown document.

        :param str text: (required), the text of the document to render
        :param str mode: (optional), 'markdown' or 'gfm'
        :param str context: (optional), only important when using mode 'gfm',
            this is the repository to use as the context for the rendering
        :param bool raw: (optional), renders a document like a README.md, no
            gfm, no context
        :returns: str -- HTML formatted text
        """
        data = None
        json = False
        headers = {}
        if raw:
            url = self._build_url('markdown', 'raw')
            data = text
            headers['content-type'] = 'text/plain'
        else:
            url = self._build_url('markdown')
            data = {}

            if text:
                data['text'] = text

            if mode in ('markdown', 'gfm'):
                data['mode'] = mode

            if context:
                data['context'] = context
            json = True

        if data:
            req = self._post(url, data=data, json=json, headers=headers)
            if req.ok:
                return req.content
        return ''