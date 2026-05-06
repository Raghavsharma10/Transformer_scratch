def create_bulk(self, indicators, user, feed):
        from .constants import API_VERSION
        if API_VERSION == '1':
            print("create_bulk currently un-avail with APIv1")
            raise SystemExit

        """
        Submit action against the IndicatorBulk endpoint

        :param indicators: list of Indicator Objects
        :param user: feed username
        :param feed: feed name
        :return: list of Indicator Objects submitted

        from csirtgsdk.client import Client
        from csirtgsdk.indicator import Indicator

        remote = 'https://csirtg.io/api'
        token = ''
        verify_ssl = True

        i = {
            'indicator': 'example.com',
            'feed': 'test',
            'user': 'admin',
            'comment': 'this is a test',
        }

        data = []

        cli = Client(remote=remote, token=token, verify_ssl=verify_ssl)

        for x in range(0, 5):
            data.append(
                Indicator(cli, i)
            )

        ret = cli.submit_bulk(data, 'csirtgadgets', 'test-feed')
        """

        uri = '/users/{0}/feeds/{1}/indicators_bulk'.format(user, feed)

        data = {
            'indicators': [
                {
                    'indicator': i.args.indicator,
                    'feed_id': i.args.feed,
                    'tag_list': i.args.tags,
                    "description": i.args.description,
                    "portlist": i.args.portlist,
                    "protocol": i.args.protocol,
                    'firsttime': i.args.firsttime,
                    'lasttime': i.args.lasttime,
                    'portlist_src': i.args.portlist_src,
                    'comment': {
                        'content': i.args.comment
                    },
                    'rdata': i.args.rdata,
                    'rtype': i.args.rtype,
                    'content': i.args.content,
                    'provider': i.args.provider,
                } for i in indicators
                ]
        }
        return self.client.post(uri, data)