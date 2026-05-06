def dir(self, path='/', slash=True, bus=False, timeout=0):
        """list entities at path"""

        if slash:
            msg = MSG_DIRALLSLASH
        else:
            msg = MSG_DIRALL
        if bus:
            flags = self.flags | FLG_BUS_RET
        else:
            flags = self.flags & ~FLG_BUS_RET

        ret, data = self.sendmess(msg, str2bytez(path), flags, timeout=timeout)
        if ret < 0:
            raise OwnetError(-ret, self.errmess[-ret], path)
        if data:
            return bytes2str(data).split(',')
        else:
            return []