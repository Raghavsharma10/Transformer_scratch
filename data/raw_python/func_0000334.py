def read_stream_stats(self):
        """
        :return: dictionary {stream index {stat name: value}}.
            Sea XenaStream.stats_captions.
        """
        stream_stats = OrderedDict()
        for stream in self.streams.values():
            stream_stats[stream] = stream.read_stats()
        return stream_stats