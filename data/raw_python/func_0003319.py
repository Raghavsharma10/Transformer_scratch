async def writelines(self, lines, eof = False, buffering = True):
        """
        Write lines to current output stream
        """
        for l in lines:
            await self.write(l, False, buffering)
        if eof:
            await self.write(b'', eof, buffering)