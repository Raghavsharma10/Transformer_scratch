def filter(self, read):
        """
        Check if a read passes the filter.

        @param read: A C{Read} instance.
        @return: C{read} if C{read} passes the filter, C{False} if not.
        """
        self.readIndex += 1

        if self.alwaysFalse:
            return False

        if self.wantedSequenceNumberGeneratorExhausted:
            return False

        if self.nextWantedSequenceNumber is not None:
            if self.readIndex + 1 == self.nextWantedSequenceNumber:
                # We want this sequence.
                try:
                    self.nextWantedSequenceNumber = next(
                        self.wantedSequenceNumberGenerator)
                except StopIteration:
                    # The sequence number iterator ran out of sequence
                    # numbers.  We must let the rest of the filtering
                    # continue for the current sequence in case we
                    # throw it out for other reasons (as we might have
                    # done for any of the earlier wanted sequence
                    # numbers).
                    self.wantedSequenceNumberGeneratorExhausted = True
            else:
                # This sequence isn't one of the ones that's wanted.
                return False

        if (self.sampleFraction is not None and
                uniform(0.0, 1.0) > self.sampleFraction):
            # Note that we don't have to worry about the 0.0 or 1.0
            # cases in the above 'if', as they have been dealt with
            # in self.__init__.
            return False

        if self.randomSubset is not None:
            if self.yieldCount == self.randomSubset:
                # The random subset has already been fully returned.
                # There's no point in going any further through the input.
                self.alwaysFalse = True
                return False
            elif uniform(0.0, 1.0) > ((self.randomSubset - self.yieldCount) /
                                      (self.trueLength - self.readIndex)):
                return False

        if self.head is not None and self.readIndex == self.head:
            # We're completely done.
            self.alwaysFalse = True
            return False

        readLen = len(read)
        if ((self.minLength is not None and readLen < self.minLength) or
                (self.maxLength is not None and readLen > self.maxLength)):
            return False

        if self.removeGaps:
            if read.quality is None:
                read = read.__class__(read.id, read.sequence.replace('-', ''))
            else:
                newSequence = []
                newQuality = []
                for base, quality in zip(read.sequence, read.quality):
                    if base != '-':
                        newSequence.append(base)
                        newQuality.append(quality)
                read = read.__class__(
                    read.id, ''.join(newSequence), ''.join(newQuality))

        if (self.titleFilter and
                self.titleFilter.accept(read.id) == TitleFilter.REJECT):
            return False

        if (self.keepSequences is not None and
                self.readIndex not in self.keepSequences):
            return False

        if (self.removeSequences is not None and
                self.readIndex in self.removeSequences):
            return False

        if self.removeDuplicates:
            if read.sequence in self.sequencesSeen:
                return False
            self.sequencesSeen.add(read.sequence)

        if self.removeDuplicatesById:
            if read.id in self.idsSeen:
                return False
            self.idsSeen.add(read.id)

        if self.modifier:
            modified = self.modifier(read)
            if modified is None:
                return False
            else:
                read = modified

        # We have to use 'is not None' in the following tests so the empty set
        # is processed properly.
        if self.keepSites is not None:
            read = read.newFromSites(self.keepSites)
        elif self.removeSites is not None:
            read = read.newFromSites(self.removeSites, exclude=True)

        if self.idLambda:
            newId = self.idLambda(read.id)
            if newId is None:
                return False
            else:
                read.id = newId

        if self.readLambda:
            newRead = self.readLambda(read)
            if newRead is None:
                return False
            else:
                read = newRead

        if self.removeDescriptions:
            read.id = read.id.split()[0]

        if self.reverse:
            read = read.reverse()
        elif self.reverseComplement:
            read = read.reverseComplement()

        self.yieldCount += 1
        return read