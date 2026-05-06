def segment(self, value=None, scope=None, metric_scope=None, **selection):
        """
        Return a new query, limited to a segment of all users or sessions.

        Accepts segment objects, filtered segment objects and segment names:

        ```python
        query.segment(account.segments['browser'])
        query.segment('browser')
        query.segment(account.segments['browser'].any('Chrome', 'Firefox'))
        ```

        Segment can also accept a segment expression when you pass
        in a `type` argument. The type argument can be either `users`
        or `sessions`. This is pretty close to the metal.

        ```python
        # will be translated into `users::condition::perUser::ga:sessions>10`
        query.segment('condition::perUser::ga:sessions>10', type='users')
        ```

        See the [Google Analytics dynamic segments documentation][segments]

        You can also use the `any`, `all`, `followed_by` and
        `immediately_followed_by` functions in this module to
        chain together segments.

        Everything about how segments get handled is still in flux.
        Feel free to propose ideas for a nicer interface on
        the [GitHub issues page][issues]

        [segments]: https://developers.google.com/analytics/devguides/reporting/core/v3/segments#reference
        [issues]: https://github.com/debrouwere/google-analytics/issues
        """

        """
        Technical note to self about segments:

        * users or sessions
        * sequence or condition
        * scope (perHit, perSession, perUser -- gte primary scope)

        Multiple conditions can be ANDed or ORed together; these two are equivalent

            users::condition::ga:revenue>10;ga:sessionDuration>60
            users::condition::ga:revenue>10;users::condition::ga:sessionDuration>60

        For sequences, prepending ^ means the first part of the sequence has to match
        the first session/hit/...

        * users and sessions conditions can be combined (but only with AND)
        * sequences and conditions can also be combined (but only with AND)

        sessions::sequence::ga:browser==Chrome;
        condition::perHit::ga:timeOnPage>5
        ->>
        ga:deviceCategory==mobile;ga:revenue>10;

        users::sequence::ga:deviceCategory==desktop
        ->>
        ga:deviceCategory=mobile;
        ga:revenue>100;
        condition::ga:browser==Chrome

        Problem: keyword arguments are passed as a dictionary, not an ordered dictionary!
        So e.g. this is risky

            query.sessions(time_on_page__gt=5, device_category='mobile', followed_by=True)
        """

        SCOPES = {
            'hits': 'perHit',
            'sessions': 'perSession',
            'users': 'perUser',
            }
        segments = self.meta.setdefault('segments', [])

        if value and len(selection):
            raise ValueError("Cannot specify a filter string and a filter keyword selection at the same time.")
        elif value:
            value = [self.api.segments.serialize(value)]
        elif len(selection):
            if not scope:
                raise ValueError("Scope is required. Choose from: users, sessions.")

            if metric_scope:
                metric_scope = SCOPES[metric_scope]

            value = select(self.api.columns, selection)
            value = [[scope, 'condition', metric_scope, condition] for condition in value]
            value = ['::'.join(filter(None, condition)) for condition in value]

        segments.append(value)
        self.raw['segment'] = utils.paste(segments, ',', ';')
        return self