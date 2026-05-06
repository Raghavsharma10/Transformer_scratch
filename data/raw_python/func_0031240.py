def stop_capture(self, cap_file_name=None, cap_file_format=IxeCapFileFormat.mem):
        """ Stop capture on port.

        :param cap_file_name: prefix for the capture file name.
            Capture file will be saved as pcap file named 'prefix' + 'URI'.pcap.
        :param cap_file_format: exported file format
        :return: number of captured frames
        """

        return self.session.stop_capture(cap_file_name, cap_file_format, self)[self]