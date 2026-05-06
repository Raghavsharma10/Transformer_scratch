def gauge(slug, maximum=9000, size=200, coerce='float'):
    """Include a Donut Chart for the specified Gauge.

    * ``slug`` -- the unique slug for the Gauge.
    * ``maximum`` -- The maximum value for the gauge (default is 9000)
    * ``size`` -- The size (in pixels) of the gauge (default is 200)
    * ``coerce`` -- type to which gauge values should be coerced. The default
      is float. Use ``{% gauge some_slug coerce='int' %}`` to coerce to integer

    """
    coerce_options = {'float': float, 'int': int, 'str': str}
    coerce = coerce_options.get(coerce, float)

    redis = get_r()
    value = coerce(redis.get_gauge(slug))
    if value < maximum and coerce == float:
        diff = round(maximum - value, 2)
    elif value < maximum:
        diff = maximum - value
    else:
        diff = 0

    return {
        'slug': slug,
        'current_value': value,
        'max_value': maximum,
        'size': size,
        'diff': diff,
    }