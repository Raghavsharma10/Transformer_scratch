def raptorize(self, resp):
        """ Raptorize this response!

        Insert javascript into the <head> tag.

        If jquery is already included, make sure not to stomp on it by
        re-including it.
        """

        soup = BeautifulSoup.BeautifulSoup(resp.body)

        if not soup.html:
            return resp

        if not soup.html.head:
            soup.html.insert(0, BeautifulSoup.Tag(soup, "head"))

        prefix = self.resources_app.prefix
        js_helper = BeautifulSoup.Tag(
            soup, "script", attrs=[
                ('type', 'text/javascript'),
                ('src', prefix + '/js_helper.js'),
            ])
        soup.html.head.insert(len(soup.html.head), js_helper)

        payload_js = BeautifulSoup.Tag(
            soup, "script", attrs=[
                ('type', 'text/javascript'),
            ])
        payload_js.setString(
            """
            run_with_jquery(function() {
                include_js("%s", function() {
                    $(window).load(function() {
                        $('body').raptorize({
                            enterOn: "%s",
                            delayTime: %i,
                        });
                    });
                })
            });
            """ % (
                prefix + '/jquery.raptorize.1.0.js',
                self.enterOn,
                self.delayTime
            )
        )
        soup.html.head.insert(len(soup.html.head), payload_js)

        resp.body = str(soup.prettify())
        return resp