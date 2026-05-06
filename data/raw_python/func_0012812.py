def tdbr2EOL(td):
    """convert the <br/> in <td> block into line ending (EOL = \n)"""
    for br in td.find_all("br"):
        br.replace_with("\n")
    txt = six.text_type(td) # make it back into test 
                            # would be unicode(id) in python2
    soup = BeautifulSoup(txt, 'lxml') # read it as a BeautifulSoup
    ntxt = soup.find('td') # BeautifulSoup has lot of other html junk.
                           # this line will extract just the <td> block 
    return ntxt