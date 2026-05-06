def digital_channel_air(self, opt1='?', opt2='?'):
        """
        Description:

            Change Channel (Digital)
            Pass Channels "XX.YY" as TV.digital_channel_air(XX, YY)

        Arguments:
            opt1: integer
                1-99: Major Channel
            opt2: integer (optional)
                1-99: Minor Channel
        """
        if opt1 == '?':
            parameter = '?'
        elif opt2 == '?':
            parameter = str(opt1).rjust(4, "0")
        else:
            parameter = '{:02d}{:02d}'.format(opt1, opt2)
        return self._send_command('digital_channel_air', parameter)