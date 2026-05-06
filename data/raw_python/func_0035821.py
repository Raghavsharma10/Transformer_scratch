def cohort_queryplan(plan):
  """
  Input:
  {
   'source': 'kronos', # Name of data source from settings
   'cohort':
    {'stream': CohortTest.EMAIL_STREAM, # Kronos stream to define cohort from.
     'transform': lambda x: x,          # Transformations on the kstream.
     'start': date.now(),               # The day of the first cohort.
     'unit': DateUnit.XX,               # Users are in the same cohort
                                        # if they are in the same day/week.
     'cohorts': 5                       # How many cohorts (days/weeks/months)
                                        # to track.
     'grouping_key': 'user'},           # What key in an event should we tie
                                        # to a key in the action stream?

   'action':
     {'stream': CohortTest.FRONTPAGE_STREAM, # Stream users take actions on.
      'transform': lambda x: x               # Transformations on the stream.
      'unit': DateUnit.XX,                   # Track events in day/week/months.
      'repetitions': 14                   # How many days/weeks/months to track.
      'grouping_key': 'user_id'}          # What key in an event should we tie
                                          # to a key in the action stream?
  }

  Output:
  A metis-compatible query plan to return a cohort analysis.
  """
  cohort = plan['cohort']
  action = plan['action']
  source = plan['source']

  # Calculate the start and end dates, in Kronos time, of the
  # beginning and end of the cohort and action streams that will be
  # relevant.
  cohort_start = datetime_to_kronos_time(_date_to_datetime(cohort['start']))
  cohort_span = timedelta(**{cohort['unit']: cohort['cohorts']})
  cohort_end = cohort['start'] + cohort_span
  action_span = timedelta(**{action['unit']: action['repetitions']})
  action_end = cohort_end + action_span
  cohort_end = datetime_to_kronos_time(_date_to_datetime(cohort_end)) + 1
  action_end = datetime_to_kronos_time(_date_to_datetime(action_end)) + 1

  left = _cohort_stream_transform(source,
                                  cohort['stream'], cohort_start, cohort_end,
                                  cohort.get('transform'),
                                  cohort['grouping_key'], cohort['unit'])
  right = _cohort_stream_transform(source,
                                   action['stream'], cohort_start, action_end,
                                   action.get('transform'),
                                   action['grouping_key'], action['unit'])

  additional_action_time = (DateUnit.unit_to_kronos_time(action['unit']) *
                            action['repetitions'])

  left.alias = 'cohort'
  right.alias = 'action'

  joined = Join(left,
                right,
                (Condition(Condition.Op.EQ,
                           Property('cohort.%s' % cohort['grouping_key']),
                           Property('action.%s' % action['grouping_key'])) &
                 Condition(Condition.Op.GTE,
                           Property('action.%s' % TIMESTAMP_FIELD),
                           Property('cohort.%s' % TIMESTAMP_FIELD)) &
                 Condition(Condition.Op.LT,
                           Property('action.%s' % TIMESTAMP_FIELD),
                           Add([Property('cohort.%s' % TIMESTAMP_FIELD),
                                Constant(additional_action_time)]))))

  user_aggregated = Aggregate(
    joined,
    GroupBy([Property('cohort.date', alias=TIMESTAMP_FIELD),
             Property('cohort.%s' % cohort['grouping_key'], alias='group'),
             Floor([Subtract([Property('action.%s' % TIMESTAMP_FIELD),
                              Property('cohort.%s' % TIMESTAMP_FIELD)]),
                    Constant(DateUnit.unit_to_kronos_time(action['unit']))],
                   alias='action_step')]),
    [Count([], alias='count')]
  )

  aggregated = Aggregate(
    user_aggregated,
    GroupBy([Property(TIMESTAMP_FIELD, alias=TIMESTAMP_FIELD),
             Property('action_step', alias='action_step')]),
    [Count([], alias='cohort_actions')])

  # TODO(marcua): Also sum up the cohort sizes, join with the plan.
  return aggregated.to_dict()