def data(tableid,
         variables=dict(),
         stream=False,
         descending=False,
         lang=DEFAULT_LANGUAGE):
    """Pulls data from a table and generates rows.

    Variables is a dictionary mapping variable codes to values.

    Streaming:
    Values must be chosen for all variables when streaming
    """
    # bulk is also in csv format, but the response is streamed
    format = 'BULK' if stream else 'CSV'

    request = Request('data', tableid, format,
                      timeOrder='Descending' if descending else None,
                      valuePresentation='CodeAndValue',
                      lang=lang,
                      **variables)

    return (Data(datum, lang=lang) for datum in request.csv)