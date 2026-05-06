def __split_name(self, name):
        u"""
        Разделяет имя на сегменты по разделителям в self.separators
        :param name: имя
        :return: разделённое имя вместе с разделителями
        """
        def gen(name, separators):
            if len(separators) == 0:
                yield name
            else:
                segments = name.split(separators[0])
                for subsegment in gen(segments[0], separators[1:]):
                    yield subsegment
                for segment in segments[1:]:
                    for subsegment in gen(segment, separators[1:]):
                        yield separators[0]
                        yield subsegment

        return gen(name, self.separators)