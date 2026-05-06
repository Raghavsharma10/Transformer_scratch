def get_article(url):
    """
    Return a single article as a 'amcat-ready' dict
    Uses the 'export' function of wikinews to get an xml article
    """
    a = html.parse(url).getroot()
    title = a.cssselect(".firstHeading")[0].text_content()
    date = a.cssselect(".published")[0].text_content()
    date = datetime.datetime.strptime(date, "%A, %B %d, %Y").isoformat()
    paras = a.cssselect("#mw-content-text p")
    paras = paras[1:] # skip first paragraph, which contains date
    text = "\n\n".join(p.text_content().strip() for p in paras)

    return dict(headline=title,
                date=date,
                url=url,
                text=text,
                medium="Wikinews")