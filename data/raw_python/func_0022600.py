def metrics_since(slugs, years, link_type="detail", granularity=None):
    """Renders a template with a menu to view a metric (or metrics) for a
    given number of years.

    * ``slugs`` -- A Slug or a set/list of slugs
    * ``years`` -- Number of years to show past metrics
    * ``link_type`` -- What type of chart do we want ("history" or "aggregate")
        * history  -- use when displaying a single metric's history
        * aggregate -- use when displaying aggregate metric history
    * ``granularity`` -- For "history" only; show the metric's granularity;
      default is "daily"

    """
    now = datetime.utcnow()

    # Determine if we're looking at one slug or multiple slugs
    if type(slugs) in [list, set]:
        slugs = "+".join(s.lower().strip() for s in slugs)

    # Set the default granularity if it's omitted
    granularity = granularity.lower().strip() if granularity else "daily"

    # Each item is: (slug, since, text, granularity)
    # Always include values for Today, 1 week, 30 days, 60 days, 90 days...
    slug_values = [
        (slugs, now - timedelta(days=1), "Today", granularity),
        (slugs, now - timedelta(days=7), "1 Week", granularity),
        (slugs, now - timedelta(days=30), "30 Days", granularity),
        (slugs, now - timedelta(days=60), "60 Days", granularity),
        (slugs, now - timedelta(days=90), "90 Days", granularity),
    ]

    # Then an additional number of years
    for y in range(1, years + 1):
        t = now - timedelta(days=365 * y)
        text = "{0} Years".format(y)
        slug_values.append((slugs, t, text, granularity))
    return {'slug_values': slug_values, 'link_type': link_type.lower().strip()}