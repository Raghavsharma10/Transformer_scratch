def line(self, x0, y0, x1, y1):
        """Draw a line using Xiaolin Wu's antialiasing technique"""
        # clean params
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if y0 > y1:
            y0, y1, x0, x1 = y1, y0, x1, x0
        dx = x1 - x0
        if dx < 0:
            sx = -1
        else:
            sx = 1
        dx *= sx
        dy = y1 - y0

        # 'easy' cases
        if dy == 0:
            for x in range(x0, x1, sx):
                self.point(x, y0)
            return
        if dx == 0:
            for y in range(y0, y1):
                self.point(x0, y)
            self.point(x1, y1)
            return
        if dx == dy:
            for x in range(x0, x1, sx):
                self.point(x, y0)
                y0 += 1
            return

        # main loop
        self.point(x0, y0)
        e_acc = 0
        if dy > dx:  # vertical displacement
            e = (dx << 16) // dy
            for i in range(y0, y1 - 1):
                e_acc_temp, e_acc = e_acc, (e_acc + e) & 0xFFFF
                if e_acc <= e_acc_temp:
                    x0 += sx
                w = 0xFF-(e_acc >> 8)
                self.point(x0, y0, intensity(self.color, w))
                y0 += 1
                self.point(x0 + sx, y0, intensity(self.color, (0xFF - w)))
            self.point(x1, y1)
            return

        # horizontal displacement
        e = (dy << 16) // dx
        for i in range(x0, x1 - sx, sx):
            e_acc_temp, e_acc = e_acc, (e_acc + e) & 0xFFFF
            if e_acc <= e_acc_temp:
                y0 += 1
            w = 0xFF-(e_acc >> 8)
            self.point(x0, y0, intensity(self.color, w))
            x0 += sx
            self.point(x0, y0 + 1, intensity(self.color, (0xFF-w)))
        self.point(x1, y1)