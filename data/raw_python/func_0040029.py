def check_battery(self):
        """
        Implement how we will check battery condition. Now it just trying to check standard battery
        in /sys
        """
        self.charging = False if \
            subprocess.getoutput("cat /sys/class/power_supply/BAT0/status") == 'Discharging' \
            else True
        percent = subprocess.getoutput("cat /sys/class/power_supply/BAT0/capacity")
        if not self.charging:
            for val in self.dischlist:
                if int(percent) <= int(val):
                    self.indicator.set_icon(self.dischformat.format(value=val))
                    break
        else:
            for val in self.chlist:
                if int(percent) <= int(val):
                    self.indicator.set_icon(self.chformat.format(value=val))
                    break
        return True