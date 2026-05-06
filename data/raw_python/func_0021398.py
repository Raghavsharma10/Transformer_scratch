def get_i2c_bus_numbers(glober = glob.glob):
        """Search all the available I2C devices in the system"""
        res = []
        for device in glober("/dev/i2c-*"):
            r = re.match("/dev/i2c-([\d]){1,2}", device)
            res.append(int(r.group(1)))
        return res