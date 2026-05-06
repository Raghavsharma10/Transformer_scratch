def is_bam_valid(bam_file):
    """Check if bam file is valid. Raises a ValueError if pysam cannot read the file.

    #TODO: pysam  does not differentiate between BAM and SAM
    """
    try:
        f = pysam.AlignmentFile(bam_file)
    except ValueError:
        raise
    except:
        raise
    else:
        f.close()

    return True