def segment(self, tokens):
        """
        Segments a sequence of tokens into a sequence of segments.

        :Parameters:
            tokens : `list` ( :class:`~deltas.Token` )
        """
        look_ahead = LookAhead(tokens)

        segments = Segment()

        while not look_ahead.empty():

            if look_ahead.peek().type not in self.whitespace:  # Paragraph!
                paragraph = MatchableSegment(look_ahead.i)

                while not look_ahead.empty() and \
                      look_ahead.peek().type not in self.paragraph_end:

                    if look_ahead.peek().type == "tab_open":  # Table
                        tab_depth = 1
                        sentence = MatchableSegment(
                            look_ahead.i, [next(look_ahead)])
                        while not look_ahead.empty() and tab_depth > 0:
                            tab_depth += look_ahead.peek().type == "tab_open"
                            tab_depth -= look_ahead.peek().type == "tab_close"
                            sentence.append(next(look_ahead))

                        paragraph.append(sentence)

                    elif look_ahead.peek().type not in self.whitespace:  # Sentence!
                        sentence = MatchableSegment(
                            look_ahead.i, [next(look_ahead)])
                        sub_depth = int(sentence[0].type in SUB_OPEN)
                        while not look_ahead.empty():

                            sub_depth += look_ahead.peek().type in SUB_OPEN
                            sub_depth -= look_ahead.peek().type in SUB_CLOSE
                            sentence.append(next(look_ahead))

                            if sentence[-1].type in self.sentence_end and sub_depth <= 0:
                                non_whitespace = sum(s.type not in self.whitespace for s in sentence)
                                if non_whitespace >= self.min_sentence:
                                    break

                        paragraph.append(sentence)

                    else:  # look_ahead.peek().type in self.whitespace
                        whitespace = Segment(look_ahead.i, [next(look_ahead)])
                        paragraph.append(whitespace)

                segments.append(paragraph)
            else: # look_ahead.peek().type in self.whitespace
                whitespace = Segment(look_ahead.i, [next(look_ahead)])
                segments.append(whitespace)


        return segments