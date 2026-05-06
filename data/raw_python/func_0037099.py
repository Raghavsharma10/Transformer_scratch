def escapePlaceholders(self,inputString):
        """
        This is an internal method that escapes all the placeholders
        defined in MapConstants.py.
        """
        escaped = inputString.replace(MapConstants.placeholder,'\\'+MapConstants.placeholder)
        escaped = escaped.replace(MapConstants.placeholderFileName,'\\'+MapConstants.placeholderFileName)
        escaped = escaped.replace(MapConstants.placeholderPath,'\\'+MapConstants.placeholderPath)
        escaped = escaped.replace(MapConstants.placeholderExtension,'\\'+MapConstants.placeholderExtension)
        escaped = escaped.replace(MapConstants.placeholderCounter,'\\'+MapConstants.placeholderCounter)
        return escaped