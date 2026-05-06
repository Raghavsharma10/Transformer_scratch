def set_html(self, html, url = None):
    """ Sets custom HTML in our Webkit session and allows to specify a fake URL.
    Scripts and CSS is dynamically fetched as if the HTML had been loaded from
    the given URL. """
    if url:
      self.conn.issue_command('SetHtml', html, url)
    else:
      self.conn.issue_command('SetHtml', html)