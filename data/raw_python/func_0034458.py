def get_traceback_html(self, **kwargs):
        "Return HTML version of debug 500 HTTP error page."
        t = Template(TECHNICAL_500_TEMPLATE)
        c = self.get_traceback_data()
        c['kwargs'] = kwargs
        return t.render(Context(c))