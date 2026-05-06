def download_release(download_file, release=None):
    """Downloads the "go-basic.obo" file for the specified release."""
    if release is None:
        release = get_latest_release()
    url = 'http://viewvc.geneontology.org/viewvc/GO-SVN/ontology-releases/%s/go-basic.obo' % release
    #download_file = 'go-basic_%s.obo' % release
    misc.http_download(url, download_file)