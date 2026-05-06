def get_packets(self, from_index=0, to_index=None, cap_type=XenaCaptureBufferType.text,
                    file_name=None, tshark=None):
        """ Get captured packets from chassis.

        :param from_index: index of first packet to read.
        :param to_index: index of last packet to read. If None - read all packets.
        :param cap_type: returned capture format. If pcap then file name and tshark must be provided.
        :param file_name: if specified, capture will be saved in file.
        :param tshark: tshark object for pcap type only.
        :type: xenamanager.xena_tshark.Tshark
        :return: list of requested packets, None for pcap type.
        """

        to_index = to_index if to_index else len(self.packets)

        raw_packets = []
        for index in range(from_index, to_index):
            raw_packets.append(self.packets[index].get_attribute('pc_packet').split('0x')[1])

        if cap_type == XenaCaptureBufferType.raw:
            self._save_captue(file_name, raw_packets)
            return raw_packets

        text_packets = []
        for raw_packet in raw_packets:
            text_packet = ''
            for c, b in zip(range(len(raw_packet)), raw_packet):
                if c % 32 == 0:
                    text_packet += '\n{:06x} '.format(int(c / 2))
                elif c % 2 == 0:
                    text_packet += ' '
                text_packet += b
            text_packets.append(text_packet)

        if cap_type == XenaCaptureBufferType.text:
            self._save_captue(file_name, text_packets)
            return text_packets

        temp_file_name = file_name + '_'
        self._save_captue(temp_file_name, text_packets)
        tshark.text_to_pcap(temp_file_name, file_name)
        os.remove(temp_file_name)