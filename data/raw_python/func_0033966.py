def show(self):
        """Redraw the text progress bar."""

        if len(self.text) > self.textwidth:
            label = self.text[0:self.textwidth]
        else:
            label = self.text.rjust(self.textwidth)

        terminalSize = getTerminalSize()
        if terminalSize is None:
            terminalSize = 80
        else:
            terminalSize = terminalSize[1]

        barWidth = terminalSize - self.textwidth - 10

        if self.value is None or self.value < 0:
            pattern = self.twiddle_sequence[
                self.twiddle % len(self.twiddle_sequence)]
            self.twiddle += 1
            barSymbols = (pattern * int(math.ceil(barWidth/3.0)))[0:barWidth]
            progressFractionText = '   . %'
        else:
            progressFraction = float(self.value) / self.max

            nBlocksFrac, nBlocksInt = math.modf(
                max(0.0, min(1.0, progressFraction)) * barWidth)
            nBlocksInt = int(nBlocksInt)

            partialBlock = self.sequence[
                int(math.floor(nBlocksFrac * len(self.sequence)))]

            nBlanks = barWidth - nBlocksInt - 1
            barSymbols = (self.sequence[-1] * nBlocksInt) + partialBlock + \
                (self.sequence[0] * nBlanks)
            barSymbols = barSymbols[:barWidth]
            progressFractionText = ('%.1f%%' % (100*progressFraction)).rjust(6)

        print >>self.fid, '\r\x1B[1m' + label + '\x1B[0m [' + barSymbols + \
            ']' + progressFractionText,
        self.fid.flush()
        self.linefed = False