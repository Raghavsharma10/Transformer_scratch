def encode2(self):
        """Return the base64 encoding of the fig attribute and insert in html image tag."""
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        string = b64encode(buf.read())
        return '<img src="data:image/png;base64,{0}">'.format(urlquote(string))