def stream(self, flags=0, devpath=None):
        "Ask gpsd to stream reports at your client."
        if (flags & (WATCH_JSON|WATCH_OLDSTYLE|WATCH_NMEA|WATCH_RAW)) == 0:
            flags |= WATCH_JSON
        if flags & WATCH_DISABLE:
            if flags & WATCH_OLDSTYLE:
                arg = "w-"
                if flags & WATCH_NMEA:
                    arg += 'r-'
                    return self.send(arg)
            else:
                gpsjson.stream(self, ~flags, devpath)
        else: # flags & WATCH_ENABLE:
            if flags & WATCH_OLDSTYLE:
                arg = 'w+'
                if (flags & WATCH_NMEA):
                    arg += 'r+'
                    return self.send(arg)
            else:
                gpsjson.stream(self, flags, devpath)