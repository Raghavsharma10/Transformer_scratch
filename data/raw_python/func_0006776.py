def data_sanitise(self, inputstring, header=None):
        """
        Format the data to be consistent with heatmaps
        :param inputstring: string containing data to be formatted
        :param header: class of the data - certain categories have specific formatting requirements
        :return: the formatted output string
        """
        if str(inputstring) == 'nan':
            outputstring = 0
        elif '%' in str(inputstring):
            group = re.findall('(\d+)\..+', str(inputstring))
            outputstring = group[0]
        elif header == 'Pass/Fail':
            if str(inputstring) == '+':
                outputstring = '100'
            else:
                outputstring = -100
                self.fail = True
        elif header == 'ContamStatus':
            if str(inputstring) == 'Clean':
                outputstring = '100'
            else:
                outputstring = -100
                self.fail = True
        elif header == 'MeanCoverage':
            cov = float(str(inputstring).split(' ')[0])
            if cov >= 20:
                outputstring = 100
            else:
                outputstring = -100
                self.fail = True
        else:
            outputstring = str(inputstring)
        return outputstring