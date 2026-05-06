async def syscall(self, func, ignoreException = False):
        """
        Call a syscall method and retrieve its return value
        """
        ev = await self.syscall_noreturn(func)
        if hasattr(ev, 'exception'):
            if ignoreException:
                return
            else:
                raise ev.exception[1]
        else:
            return ev.retvalue