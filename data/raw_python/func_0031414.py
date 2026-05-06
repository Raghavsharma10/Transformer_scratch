def toDict(self):
        """
        Get information about the title's alignments as a dictionary.

        @return: A C{dict} representation of the title's aligments.
        """
        return {
            'titleAlignments': [titleAlignment.toDict()
                                for titleAlignment in self],
            'subjectTitle': self.subjectTitle,
            'subjectLength': self.subjectLength,
        }