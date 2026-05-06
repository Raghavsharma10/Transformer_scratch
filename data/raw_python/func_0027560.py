def from_page_xml(cls, page_xml):
        """
        Constructs a :class:`~mwxml.iteration.dump.Dump` from a <page> block.

        :Parameters:
            page_xml : `str` | `file`
                Either a plain string or a file containing <page> block XML to
                process
        """
        header = """
        <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.5/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.5/
                     http://www.mediawiki.org/xml/export-0.5.xsd" version="0.5"
                   xml:lang="en">
        <siteinfo>
        </siteinfo>
        """

        footer = "</mediawiki>"

        return cls.from_file(mwtypes.files.concat(header, page_xml, footer))