def grant_client(self, client_id, publish=False, subscribe=False, publish_protocol=None, publish_topics=None,
                     subscribe_topics=None, scope_prefix='predix-event-hub', **kwargs):
        """
        Grant the given client id all the scopes and authorities
        needed to work with the eventhub service.
        """
        scopes = ['openid']
        authorities = ['uaa.resource']

        zone_id = self.get_zone_id()
        # always must be part of base user scope
        scopes.append('%s.zones.%s.user' % (scope_prefix, zone_id))
        authorities.append('%s.zones.%s.user' % (scope_prefix, zone_id))

        if publish_topics is not None or subscribe_topics is not None:
            raise Exception("multiple topics are not currently available in preidx-py")

        if publish_topics is None:
            publish_topics = ['topic']

        if subscribe_topics is None:
            subscribe_topics = ['topic']

        if publish:
            # we are granting just the default topic
            if publish_protocol is None:
                scopes.append('%s.zones.%s.grpc.publish' % (scope_prefix, zone_id))
                authorities.append('%s.zones.%s.grpc.publish' % (scope_prefix, zone_id))
                scopes.append('%s.zones.%s.wss.publish' % (scope_prefix, zone_id))
                authorities.append('%s.zones.%s.wss.publish' % (scope_prefix, zone_id))

            else:
                scopes.append('%s.zones.%s.%s.publish' % (scope_prefix, zone_id, publish_protocol))
                authorities.append('%s.zones.%s.%s.publish' % (scope_prefix, zone_id, publish_protocol))

            # we are requesting multiple topics
            for topic in publish_topics:
                if publish_protocol is None:
                    scopes.append('%s.zones.%s.%s.grpc.publish' % (scope_prefix, zone_id, topic))
                    scopes.append('%s.zones.%s.%s.wss.publish' % (scope_prefix, zone_id, topic))
                    scopes.append('%s.zones.%s.%s.user' % (scope_prefix, zone_id, topic))
                    authorities.append('%s.zones.%s.%s.grpc.publish' % (scope_prefix, zone_id, topic))
                    authorities.append('%s.zones.%s.%s.wss.publish' % (scope_prefix, zone_id, topic))
                    authorities.append('%s.zones.%s.%s.user' % (scope_prefix, zone_id, topic))
                else:
                    scopes.append('%s.zones.%s.%s.%s.publish' % (scope_prefix, zone_id, topic, publish_protocol))
                    authorities.append('%s.zones.%s.%s.%s.publish' % (scope_prefix, zone_id, topic, publish_protocol))
        if subscribe:
            # we are granting just the default topic
            scopes.append('%s.zones.%s.grpc.subscribe' % (scope_prefix, zone_id))
            authorities.append('%s.zones.%s.grpc.subscribe' % (scope_prefix, zone_id))

            # we are requesting multiple topics
            for topic in subscribe_topics:
                scopes.append('%s.zones.%s.%s.grpc.subscribe' % (scope_prefix, zone_id, topic))
                authorities.append('%s.zones.%s.%s.grpc.subscribe' % (scope_prefix, zone_id, topic))

        self.service.uaa.uaac.update_client_grants(client_id, scope=scopes,
                                                   authorities=authorities)

        return self.service.uaa.uaac.get_client(client_id)