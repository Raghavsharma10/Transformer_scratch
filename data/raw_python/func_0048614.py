def _get_parameterized_text(self, parameters):
        """stub"""
        text = self.get_text('edxml').text
        done = False
        while not done:
            result = re.search(r'\$\w+', text)
            if result:
                replacement = str(parameters[result.group()[1:]])
                text = text.replace(result.group(), replacement)
            else:
                done = True
        return text