async def copy_to(self, dest, container, buffering = True):
        """
        Coroutine method to copy content from this stream to another stream.
        """
        if self.eof:
            await dest.write(u'' if self.isunicode else b'', True)
        elif self.errored:
            await dest.error(container)
        else:
            try:
                while not self.eof:
                    await self.prepareRead(container)
                    data = self.readonce()
                    try:
                        await dest.write(data, container, self.eof, buffering = buffering)
                    except IOError:
                        break
            except:
                async def _cleanup():
                    try:
                        await dest.error(container)
                    except IOError:
                        pass
                container.subroutine(_cleanup(), False)
                raise
            finally:
                self.close(container.scheduler)