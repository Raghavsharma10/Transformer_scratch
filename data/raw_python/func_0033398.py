def to_string(self):
        """Export this namespace to a string suitable for incorporation
        in a VW example line, e.g.
        'MetricFeatures:3.28 height:1.5 length:2.0 '
        """
        if self._string is None:
            tokens = []
            if self.name:
                if self.scale:
                    token = self.name + ':' + str(self.scale)
                else:
                    token = self.name
            else:
                token = ''  # Spacing element to indicate next string is a feature
            tokens.append(token)
            for label, value in self.features:
                if value is None:
                    token = label
                else:
                    token = label + ':' + str(value)
                tokens.append(token)
            tokens.append('')  # Spacing element to separate from next pipe character
            output = ' '.join(tokens)
            if self.cache_string:
                self._string = output
        else:
            output = self._string
        return output