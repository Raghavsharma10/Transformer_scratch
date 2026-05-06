def to_row(advice):
    '''Serialize an advice into a CSV row'''
    return [
        advice.id,
        advice.administration,
        advice.type,
        advice.session.year,
        advice.session.strftime('%d/%m/%Y'),
        advice.subject,
        ', '.join(advice.topics),
        ', '.join(advice.tags),
        ', '.join(advice.meanings),
        ROMAN_NUMS.get(advice.part, ''),
        advice.content,
    ]