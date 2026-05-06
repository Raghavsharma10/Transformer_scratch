def convertReadAlignment(self, read, readGroupSet, readGroupId):
        """
        Convert a pysam ReadAlignment to a GA4GH ReadAlignment
        """
        samFile = self.getFileHandle(self._dataUrl)
        # TODO fill out remaining fields
        # TODO refine in tandem with code in converters module
        ret = protocol.ReadAlignment()
        # ret.fragmentId = 'TODO'
        ret.aligned_quality.extend(read.query_qualities)
        ret.aligned_sequence = read.query_sequence
        if SamFlags.isFlagSet(read.flag, SamFlags.READ_UNMAPPED):
            ret.ClearField("alignment")
        else:
            ret.alignment.CopyFrom(protocol.LinearAlignment())
            ret.alignment.mapping_quality = read.mapping_quality
            ret.alignment.position.CopyFrom(protocol.Position())
            ret.alignment.position.reference_name = samFile.getrname(
                read.reference_id)
            ret.alignment.position.position = read.reference_start
            ret.alignment.position.strand = protocol.POS_STRAND
            if SamFlags.isFlagSet(read.flag, SamFlags.READ_REVERSE_STRAND):
                ret.alignment.position.strand = protocol.NEG_STRAND
            for operation, length in read.cigar:
                gaCigarUnit = ret.alignment.cigar.add()
                gaCigarUnit.operation = SamCigar.int2ga(operation)
                gaCigarUnit.operation_length = length
                gaCigarUnit.reference_sequence = ""  # TODO fix this!
        ret.duplicate_fragment = SamFlags.isFlagSet(
            read.flag, SamFlags.DUPLICATE_READ)
        ret.failed_vendor_quality_checks = SamFlags.isFlagSet(
            read.flag, SamFlags.FAILED_QUALITY_CHECK)
        ret.fragment_length = read.template_length
        ret.fragment_name = read.query_name
        for key, value in read.tags:
            # Useful for inspecting the structure of read tags
            # print("{key} {ktype}: {value}, {vtype}".format(
            #     key=key, ktype=type(key), value=value, vtype=type(value)))
            protocol.setAttribute(ret.attributes.attr[key].values, value)

        if SamFlags.isFlagSet(read.flag, SamFlags.MATE_UNMAPPED):
            ret.next_mate_position.Clear()
        else:
            ret.next_mate_position.Clear()
            if read.next_reference_id != -1:
                ret.next_mate_position.reference_name = samFile.getrname(
                    read.next_reference_id)
            else:
                ret.next_mate_position.reference_name = ""
            ret.next_mate_position.position = read.next_reference_start
            ret.next_mate_position.strand = protocol.POS_STRAND
            if SamFlags.isFlagSet(read.flag, SamFlags.MATE_REVERSE_STRAND):
                ret.next_mate_position.strand = protocol.NEG_STRAND
        if SamFlags.isFlagSet(read.flag, SamFlags.READ_PAIRED):
            ret.number_reads = 2
        else:
            ret.number_reads = 1
        ret.read_number = -1
        if SamFlags.isFlagSet(read.flag, SamFlags.FIRST_IN_PAIR):
            if SamFlags.isFlagSet(read.flag, SamFlags.SECOND_IN_PAIR):
                ret.read_number = 2
            else:
                ret.read_number = 0
        elif SamFlags.isFlagSet(read.flag, SamFlags.SECOND_IN_PAIR):
            ret.read_number = 1
        ret.improper_placement = not SamFlags.isFlagSet(
            read.flag, SamFlags.READ_PROPER_PAIR)
        ret.read_group_id = readGroupId
        ret.secondary_alignment = SamFlags.isFlagSet(
            read.flag, SamFlags.SECONDARY_ALIGNMENT)
        ret.supplementary_alignment = SamFlags.isFlagSet(
            read.flag, SamFlags.SUPPLEMENTARY_ALIGNMENT)
        ret.id = readGroupSet.getReadAlignmentId(ret)
        return ret