def oneday_weather_forecast(
        location='Portland, OR',
        inputs=('Min Temperature', 'Mean Temperature', 'Max Temperature', 'Max Humidity', 'Mean Humidity', 'Min Humidity', 'Max Sea Level Pressure', 'Mean Sea Level Pressure', 'Min Sea Level Pressure', 'Wind Direction'),
        outputs=('Min Temperature', 'Mean Temperature', 'Max Temperature', 'Max Humidity'),
        date=None,
        epochs=200,
        delays=(1, 2, 3, 4),
        num_years=4,
        use_cache=False,
        verbosity=1,
        ):
    """ Provide a weather forecast for tomorrow based on historical weather at that location """
    date = make_date(date or datetime.datetime.now().date())
    num_years = int(num_years or 10)
    years = range(date.year - num_years, date.year + 1)
    df = weather.daily(location, years=years, use_cache=use_cache, verbosity=verbosity).sort()
    # because up-to-date weather history was cached above, can use that cache, regardless of use_cache kwarg
    trainer, df = train_weather_predictor(
        location,
        years=years,
        delays=delays,
        inputs=inputs,
        outputs=outputs,
        epochs=epochs,
        verbosity=verbosity,
        use_cache=True,
        )
    nn = trainer.module
    forecast = {'trainer': trainer}

    yesterday = dict(zip(outputs, nn.activate(trainer.ds['input'][-2])))
    forecast['yesterday'] = update_dict(yesterday, {'date': df.index[-2].date()})

    today = dict(zip(outputs, nn.activate(trainer.ds['input'][-1])))
    forecast['today'] = update_dict(today, {'date': df.index[-1].date()})

    ds = util.input_dataset_from_dataframe(df[-max(delays):], delays=delays, inputs=inputs, normalize=False, verbosity=0)
    tomorrow = dict(zip(outputs, nn.activate(ds['input'][-1])))
    forecast['tomorrow'] = update_dict(tomorrow, {'date': (df.index[-1] + datetime.timedelta(1)).date()})

    return forecast