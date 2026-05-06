def get_value_display(self):
        """Human friendly value output"""
        if self.display_as == 'percentage':
            return '{0}%'.format(self.latest_value)
        if self.display_as == 'boolean':
            return bool(self.latest_value)
        if self.display_as == 'byte':
            return defaultfilters.filesizeformat(self.latest_value)
        if self.display_as == 'second':
            return time.strftime('%H:%M:%S', time.gmtime(self.latest_value))
        return self.latest_value