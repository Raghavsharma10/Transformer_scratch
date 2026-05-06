def sendcommand(self, cmd, opt=None):
        "Send a telnet command (IAC)"
        if cmd in [DO, DONT]:
            if not self.DOOPTS.has_key(opt):
                self.DOOPTS[opt] = None
            if (((cmd == DO) and (self.DOOPTS[opt] != True))
            or ((cmd == DONT) and (self.DOOPTS[opt] != False))):
                self.DOOPTS[opt] = (cmd == DO)
                self.writecooked(IAC + cmd + opt)
        elif cmd in [WILL, WONT]:
            if not self.WILLOPTS.has_key(opt):
                self.WILLOPTS[opt] = ''
            if (((cmd == WILL) and (self.WILLOPTS[opt] != True))
            or ((cmd == WONT) and (self.WILLOPTS[opt] != False))):
                self.WILLOPTS[opt] = (cmd == WILL)
                self.writecooked(IAC + cmd + opt)
        else:
            self.writecooked(IAC + cmd)