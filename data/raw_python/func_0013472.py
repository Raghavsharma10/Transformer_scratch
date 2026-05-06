def runAddAnnouncement(self, flaskrequest):
        """
        Takes a flask request from the frontend and attempts to parse
        into an AnnouncePeerRequest. If successful, it will log the
        announcement to the `announcement` table with some other metadata
        gathered from the request.
        """
        announcement = {}
        # We want to parse the request ourselves to collect a little more
        # data about it.
        try:
            requestData = protocol.fromJson(
                flaskrequest.get_data(), protocol.AnnouncePeerRequest)
            announcement['hostname'] = flaskrequest.host_url
            announcement['remote_addr'] = flaskrequest.remote_addr
            announcement['user_agent'] = flaskrequest.headers.get('User-Agent')
        except AttributeError:
            # Sometimes in testing we will send protocol requests instead
            # of flask requests and so the hostname and user agent won't
            # be present.
            try:
                requestData = protocol.fromJson(
                    flaskrequest, protocol.AnnouncePeerRequest)
            except Exception as e:
                raise exceptions.InvalidJsonException(e)
        except Exception as e:
            raise exceptions.InvalidJsonException(e)

        # Validate the url before accepting the announcement
        peer = datamodel.peers.Peer(requestData.peer.url)
        peer.setAttributesJson(protocol.toJson(
                requestData.peer.attributes))
        announcement['url'] = peer.getUrl()
        announcement['attributes'] = peer.getAttributes()
        try:
            self.getDataRepository().insertAnnouncement(announcement)
        except:
            raise exceptions.BadRequestException(announcement['url'])
        return protocol.toJson(
            protocol.AnnouncePeerResponse(success=True))