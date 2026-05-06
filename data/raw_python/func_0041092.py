def join_accesses(unique_id, accesses, from_date, until_date, dayly_granularity):
    """
    Esse metodo recebe 1 ou mais chaves para um documento em específico para que
    os acessos sejam recuperados no Ratchet e consolidados em um unico id.
    Esse processo é necessário pois os acessos de um documento podem ser registrados
    para os seguintes ID's (PID, PID FBPE, Path PDF).
    PID: Id original do SciELO ex: S0102-67202009000300001
    PID FBPE: Id antigo do SciELO ex: S0102-6720(09)000300001
    Path PDF: Quando o acesso é feito diretamente para o arquivo PDF no FS do
    servidor ex: /pdf/rsp/v12n10/v12n10.pdf
    """
    logger.debug('joining accesses for: %s' % unique_id)
    joined_data = {}
    listed_data = []
    def joining_monthly(joined_data, atype, data):

        if 'total' in data:
            del(data['total'])

        for year, months in data.items():
            del(months['total'])
            for month in months:

                dt = '%s-%s' % (year[1:], month[1:])
                if not dt >= from_date[:7] or not dt <= until_date[:7]:
                    continue
                joined_data.setdefault(dt, {})
                joined_data[dt].setdefault(atype, 0)
                joined_data[dt][atype] += data[year][month]['total']

        return joined_data

    def joining_dayly(joined_data, atype, data):

        if 'total' in data:
            del(data['total'])

        for year, months in data.items():
            del(months['total'])
            for month, days in months.items():
                del(days['total'])
                for day in days:
                    dt = '%s-%s-%s' % (year[1:], month[1:], day[1:])
                    if not dt >= from_date or not dt <= until_date:
                        continue
                    joined_data.setdefault(dt, {})
                    joined_data[dt].setdefault(atype, 0)
                    joined_data[dt][atype] += data[year][month][day]

        return joined_data

    joining = joining_monthly
    if dayly_granularity:
        joining = joining_dayly

    for data in accesses:
        for key, value in data.items():
            if not key in ['abstract', 'html', 'pdf', 'readcube']:
                continue
            joined_data = joining(joined_data, key, value)

    return joined_data