def _fabtabular():
    """
    This function retrieves the ISO 639 and inverted names datasets as tsv files and returns them as lists.
    """
    import csv
    import sys
    from pkg_resources import resource_filename

    data = resource_filename(__package__, 'iso-639-3.tab')
    inverted = resource_filename(__package__, 'iso-639-3_Name_Index.tab')
    macro = resource_filename(__package__, 'iso-639-3-macrolanguages.tab')
    part5 = resource_filename(__package__, 'iso639-5.tsv')
    part2 = resource_filename(__package__, 'iso639-2.tsv')
    part1 = resource_filename(__package__, 'iso639-1.tsv')

    # if sys.version_info[0] == 2:
    #     from urllib2 import urlopen
    #     from contextlib import closing
    #     data_fo = closing(urlopen('http://www-01.sil.org/iso639-3/iso-639-3.tab'))
    #     inverted_fo = closing(urlopen('http://www-01.sil.org/iso639-3/iso-639-3_Name_Index.tab'))
    # else:
    #     from urllib.request import urlopen
    #     import io
    #     data_fo = io.StringIO(urlopen('http://www-01.sil.org/iso639-3/iso-639-3.tab').read().decode())
    #     inverted_fo = io.StringIO(urlopen('http://www-01.sil.org/iso639-3/iso-639-3_Name_Index.tab').read().decode())

    if sys.version_info[0] == 3:
        from functools import partial

        global open
        open = partial(open, encoding='utf-8')

    data_fo = open(data)
    inverted_fo = open(inverted)
    macro_fo = open(macro)
    part5_fo = open(part5)
    part2_fo = open(part2)
    part1_fo = open(part1)
    with data_fo as u:
        with inverted_fo as i:
            with macro_fo as m:
                with part5_fo as p5:
                    with part2_fo as p2:
                        with part1_fo as p1:
                            return (list(csv.reader(u, delimiter='\t'))[1:],
                                    list(csv.reader(i, delimiter='\t'))[1:],
                                    list(csv.reader(m, delimiter='\t'))[1:],
                                    list(csv.reader(p5, delimiter='\t'))[1:],
                                    list(csv.reader(p2, delimiter='\t'))[1:],
                                    list(csv.reader(p1, delimiter='\t'))[1:])