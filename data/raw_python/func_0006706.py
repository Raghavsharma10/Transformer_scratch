def process_request(self, request):
        """Processes a request."""
        try:
            request = Request.from_json(request.read().decode())
        except ValueError:
            raise ClientError('Data is not valid JSON.')
        except KeyError:
            raise ClientError('Missing mandatory field in request object.')
        except AttributeNotProvided as exc:
            raise ClientError('Attribute not provided: %s.' % exc.args[0])

        (start_wall_time, start_process_time) = self._get_times()
        answers = self.router_class(request).answer()
        self._add_times_to_answers(answers, start_wall_time, start_process_time)

        answers = [x.as_dict() for x in answers]
        return self.make_response('200 OK',
                                  'application/json',
                                  json.dumps(answers)
                                 )