def checkReference(self, reference):
        """
        Check the reference for security. Tries to avoid any characters
        necessary for doing a script injection.
        """
        pattern = re.compile(r'[\s,;"\'&\\]')
        if pattern.findall(reference.strip()):
            return False
        return True