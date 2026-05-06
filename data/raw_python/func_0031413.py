def summary(self):
        """
        Summarize the alignments for this subject.

        @return: A C{dict} with C{str} keys:
            bestScore: The C{float} best score of the matching reads.
            coverage: The C{float} fraction of the subject genome that is
                matched by at least one read.
            hspCount: The C{int} number of hsps that match the subject.
            medianScore: The C{float} median score of the matching reads.
            readCount: The C{int} number of reads that match the subject.
            subjectLength: The C{int} length of the subject.
            subjectTitle: The C{str} title of the subject.
        """
        return {
            'bestScore': self.bestHsp().score.score,
            'coverage': self.coverage(),
            'hspCount': self.hspCount(),
            'medianScore': self.medianScore(),
            'readCount': self.readCount(),
            'subjectLength': self.subjectLength,
            'subjectTitle': self.subjectTitle,
        }