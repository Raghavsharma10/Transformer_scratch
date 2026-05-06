async def error(self, status=500, allowredirect = True, close = True, showerror = None, headers = []):
        """
        Show default error response
        """
        if showerror is None:
            showerror = self.showerrorinfo
        if self._sendHeaders:
            if showerror:
                typ, exc, tb = sys.exc_info()
                if exc:
                    await self.write('<span style="white-space:pre-wrap">\n', buffering = False)
                    await self.writelines((self.nl2br(self.escape(v)) for v in traceback.format_exception(typ, exc, tb)), buffering = False)
                    await self.write('</span>\n', close, False)
        elif allowredirect and status in self.protocol.errorrewrite:
            await self.rewrite(self.protocol.errorrewrite[status], b'GET')
        elif allowredirect and status in self.protocol.errorredirect:
            await self.redirect(self.protocol.errorredirect[status])
        else:
            self.start_response(status, headers)
            typ, exc, tb = sys.exc_info()
            if showerror and exc:
                await self.write('<span style="white-space:pre-wrap">\n', buffering = False)
                await self.writelines((self.nl2br(self.escape(v)) for v in traceback.format_exception(typ, exc, tb)), buffering = False)
                await self.write('</span>\n', close, False)
            else:
                await self.write(b'<h1>' + _createstatus(status) + b'</h1>', close, False)