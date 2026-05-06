def compute(chart):
    """ Computes the Almutem table. """
    almutems = {}
    
    # Hylegic points
    hylegic = [
        chart.getObject(const.SUN),
        chart.getObject(const.MOON),
        chart.getAngle(const.ASC),
        chart.getObject(const.PARS_FORTUNA),
        chart.getObject(const.SYZYGY)
    ]
    for hyleg in hylegic:
        row = newRow()
        digInfo = essential.getInfo(hyleg.sign, hyleg.signlon)
        
        # Add the scores of each planet where hyleg has dignities
        for dignity in DIGNITY_LIST:
            objID = digInfo[dignity]
            if objID:
                score = essential.SCORES[dignity]
                row[objID]['string'] += '+%s' % score
                row[objID]['score'] += score
                
        almutems[hyleg.id] = row
        
    # House positions
    row = newRow()
    for objID in OBJECT_LIST:
        obj = chart.getObject(objID)
        house = chart.houses.getObjectHouse(obj)
        score = HOUSE_SCORES[house.id]
        row[objID]['string'] = '+%s' % score
        row[objID]['score'] = score
    almutems['Houses'] = row
    
    # Planetary time
    row = newRow()
    table = planetarytime.getHourTable(chart.date, chart.pos)
    ruler = table.currRuler()
    hourRuler = table.hourRuler()
    row[ruler] = {
        'string': '+7',
        'score': 7
    }
    row[hourRuler] = {
        'string': '+6',
        'score': 6
    }
    almutems['Rulers'] = row;
    
    # Compute scores
    scores = newRow()
    for _property, _list in almutems.items():
        for objID, values in _list.items():
            scores[objID]['string'] += values['string']
            scores[objID]['score'] += values['score']
    almutems['Score'] = scores
    
    return almutems