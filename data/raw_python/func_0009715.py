def pdftoxml(pdfdata, options=""):
    """converts pdf file to xml file"""
    pdffout = tempfile.NamedTemporaryFile(suffix='.pdf')
    pdffout.write(pdfdata)
    pdffout.flush()

    xmlin = tempfile.NamedTemporaryFile(mode='r', suffix='.xml')
    tmpxml = xmlin.name  # "temph.xml"
    cmd = 'pdftohtml -xml -nodrm -zoom 1.5 -enc UTF-8 -noframes %s "%s" "%s"' % (
        options, pdffout.name, os.path.splitext(tmpxml)[0])
    # can't turn off output, so throw away even stderr yeuch
    cmd = cmd + " >/dev/null 2>&1"
    os.system(cmd)

    pdffout.close()
    #xmlfin = open(tmpxml)
    xmldata = xmlin.read()
    xmlin.close()
    return xmldata.decode('utf-8')