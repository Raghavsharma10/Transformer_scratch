def api2_formula(req):
    """
A simple `GET`-, URL-based API to OpenFisca, making the assumption of computing formulas for a single person.

Combination
-----------

You can compute several formulas at once by combining the paths and joining them with `+`.

Example:
```
/salaire_super_brut+salaire_net_a_payer?salaire_de_base=1440
```

This will compute both `salaire_super_brut` and `salaire_net_a_payer` in a single request.

Reforms
-----------

Reforms can be requested to patch the simulation system.
To keep this endpoint URL simple, they are requested as a list in a custom HTTP header.
```
X-OpenFisca-Extensions: de_net_a_brut, landais_piketty_saez
```
This header is of course optional.


URL size limit
--------------

Using combination with a lot of parameters may lead to long URLs.
If used within the browser, make sure the resulting URL is kept
[under 2047 characters](http://stackoverflow.com/questions/417142)
for cross-browser compatibility, by splitting combined requests.
On a server, just test what your library handles.
"""
    API_VERSION = '2.1.0'
    wsgihelpers.track(req.url.decode('utf-8'))
    params = dict(req.GET)
    data = dict()

    try:
        extensions_header = req.headers.get('X-Openfisca-Extensions')

        tax_benefit_system = model.get_cached_composed_reform(
            reform_keys = extensions_header.split(','),
            tax_benefit_system = model.tax_benefit_system,
            ) if extensions_header is not None else model.tax_benefit_system

        params = normalize(params, tax_benefit_system)
        formula_names = req.urlvars.get('names').split('+')

        data['values'] = dict()
        data['period'] = parse_period(req.urlvars.get('period'))

        simulation = create_simulation(params, data['period'], tax_benefit_system)

        for formula_name in formula_names:
            column = get_column_from_formula_name(formula_name, tax_benefit_system)
            data['values'][formula_name] = compute(column.name, simulation)

    except Exception as error:
        if isinstance(error.args[0], dict):  # we raised it ourselves, in this controller
            error = error.args[0]
        else:
            error = dict(
                message = unicode(error),
                code = 500
                )

        data['error'] = error
    finally:
        return respond(req, API_VERSION, data, params)