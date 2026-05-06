def write_smet(filename, data, metadata, nodata_value=-999, mode='h', check_nan=True):
    """writes smet files

    Parameters
    ----
    filename :    filename/loction of output
    data :        data to write as pandas df
    metadata:     header to write input as dict
    nodata_value: Nodata Value to write/use
    mode:         defines if to write daily ("d") or continuos data (default 'h')
    check_nan:    will check if only nans in data and if true will not write this colums (default True)
    """

    # dictionary
    # based on smet spec V.1.1 and selfdefined
    # daily data
    dict_d=   {'tmean':'TA',
               'tmin':'TMAX',   #no spec
               'tmax':'TMIN',   #no spec
               'precip':'PSUM',
               'glob':'ISWR',     #no spec
               'hum':'RH',
               'wind':'VW'
                }

    #hourly data
    dict_h=   {'temp':'TA',
               'precip':'PSUM',
               'glob':'ISWR',     #no spec
               'hum':'RH',
               'wind':'VW'
                }
                
    #rename columns
    if mode == "d":
        data = data.rename(columns=dict_d)
    if mode == "h":
        data = data.rename(columns=dict_h)

    if check_nan:     
        #get all colums with data
        datas_in = data.sum().dropna().to_frame().T
        #get colums with no datas
        drop = [data_nan for data_nan in data.columns if data_nan not in datas_in]    
        #delete columns
        data = data.drop(drop, axis=1)
    
    with open(filename, 'w') as f:

        #preparing data
        #converte date_times to SMET timestamps
        if mode == "d":
            t = '%Y-%m-%dT00:00'
        if mode == "h":
            t = '%Y-%m-%dT%H:%M'

        data['timestamp'] = [d.strftime(t) for d in data.index]
        
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]


        #metadatas update
        metadata['fields'] = ' '.join(data.columns)
        metadata["units_multiplier"] = len(metadata['fields'].split())*"1 "

        #writing data
        #metadata
        f.write('SMET 1.1 ASCII\n')
        f.write('[HEADER]\n')

        for k, v in metadata.items():
            f.write('{} = {}\n'.format(k, v))

        #data
        f.write('[DATA]\n')

        data_str = data.fillna(nodata_value).to_string(
            header=False,
            index=False,
            float_format=lambda x: '{:.2f}'.format(x),
        )

        f.write(data_str)