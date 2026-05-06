def from_row(row):
    '''Create an advice from a CSV row'''
    subject = (row[5][0].upper() + row[5][1:]) if row[5] else row[5]
    return Advice.objects.create(
        id=row[0],
        administration=cleanup(row[1]),
        type=row[2],
        session=datetime.strptime(row[4], '%d/%m/%Y'),
        subject=cleanup(subject),
        topics=[t.title() for t in cleanup(row[6]).split(', ')],
        tags=[tag.strip() for tag in row[7].split(',') if tag.strip()],
        meanings=cleanup(row[8]).replace(' / ', '/').split(', '),
        part=_part(row[9]),
        content=cleanup(row[10]),
    )