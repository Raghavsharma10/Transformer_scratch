def analyze_xml(xml):
    """Analyzes `file` against packtools' XMLValidator.
    """

    f = StringIO(xml)

    try:
        xml = packtools.XMLValidator.parse(f, sps_version='sps-1.4')
    except packtools.exceptions.PacktoolsError as e:
        logger.exception(e)
        summary = {}
        summary['dtd_is_valid'] = False
        summary['sps_is_valid'] = False
        summary['is_valid'] = False
        summary['parsing_error'] = True
        summary['dtd_errors'] = []
        summary['sps_errors'] = []
        return summary
    except XMLSyntaxError as e:
        logger.exception(e)
        summary = {}
        summary['dtd_is_valid'] = False
        summary['sps_is_valid'] = False
        summary['is_valid'] = False
        summary['parsing_error'] = True
        summary['dtd_errors'] = [e.message]
        summary['sps_errors'] = []
        return summary
    else:
        summary = summarize(xml)

        return summary