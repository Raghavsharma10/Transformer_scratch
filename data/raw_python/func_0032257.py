def renderHTTP(self, ctx):
        """
        Extract the data from the I{uploaddata} field of the request and pass
        it to our callback.
        """
        req = inevow.IRequest(ctx)
        if req.method == 'POST':
            udata = req.fields['uploaddata']
            self.cbGotMugshot(udata.type.decode('ascii'), udata.file)
        return rend.Page.renderHTTP(self, ctx)