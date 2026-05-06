def modify(self, **kwargs):
        """Modify settings for a check. The provided settings will overwrite
            previous values. Settings not provided will stay the same as before
            the update. To clear an existing value, provide an empty value.
            Please note that you cannot change the type of a check once it has
            been created.

        General parameters:

            * name -- Check name
                    Type: String

            * host - Target host
                    Type: String

            * paused -- Check should be paused
                    Type: Boolean

            * resolution -- Check resolution time (in minutes)
                    Type: Integer [1, 5, 15, 30, 60]

            * contactids -- Comma separated list of contact IDs
                    Type: String

            * sendtoemail -- Send alerts as email
                    Type: Boolean

            * sendtosms -- Send alerts as SMS
                    Type: Boolean

            * sendtotwitter -- Send alerts through Twitter
                    Type: Boolean

            * sendtoiphone -- Send alerts to iPhone
                    Type: Boolean

            * sendtoandroid -- Send alerts to Android
                    Type: Boolean

            * sendnotificationwhendown -- Send notification when check is down
                the given number of times
                    Type: Integer

            * notifyagainevery -- Set how many results to wait for in between
                notices
                    Type: Integer

            * notifywhenbackup -- Notify when back up again
                    Type: Boolean

            * use_legacy_notifications -- Use old notifications instead of BeepManager
                    Type: Boolean

            * probe_filters -- Can be one of region: NA, region: EU, region: APAC
                    Type: String

        HTTP check options:

            * url -- Target path on server
                    Type: String

            * encryption -- Use SSL/TLS
                    Type: Boolean

            * port -- Target server port
                    Type: Integer

            * auth -- Username and password for HTTP authentication
                Example: user:password
                    Type: String

            * shouldcontain -- Target site should contain this string.
                Cannot be combined with 'shouldnotcontain'
                    Type: String

            * shouldnotcontain -- Target site should not contain this string.
                Cannot be combined with 'shouldcontain'
                    Type: String

            * postdata -- Data that should be posted to the web page,
                for example submission data for a sign-up or login form.
                The data needs to be formatted in the same way as a web browser
                would send it to the web server
                    Type: String

            * requestheader<NAME> -- Custom HTTP header, replace <NAME> with
                desired header name. Header in form: Header:Value
                    Type: String

        HTTPCustom check options:

            * url -- Target path on server
                    Type: String

            * encryption -- Use SSL/TLS
                    Type: Boolean

            * port -- Target server port
                    Type: Integer

            * auth -- Username and password for HTTP authentication
                Example: user:password
                    Type: String

            * additionalurls -- Colon-separated list of additonal URLS with
                hostname included
                    Type: String

        TCP check options:

            * port -- Target server port
                    Type: Integer

            * stringtosend -- String to send
                    Type: String

            * stringtoexpect -- String to expect in response
                    Type: String

        DNS check options:

            * expectedip -- Expected IP
                    Type: String

            * nameserver -- Nameserver to check
                    Type: String

        UDP check options:

            * port -- Target server port
                    Type: Integer

            * stringtosend -- String to send
                    Type: String

            * stringtoexpect -- String to expect in response
                    Type: String

        SMTP check options:

            * port -- Target server port
                    Type: Integer

            * auth -- Username and password for target SMTP authentication.
                Example: user:password
                    Type: String

            * stringtoexpect -- String to expect in response
                    Type: String

            * encryption -- Use connection encryption
                    Type: Boolean

        POP3 check options:

            * port -- Target server port
                    Type: Integer

            * stringtoexpect -- String to expect in response
                    Type: String

            * encryption -- Use connection encryption
                    Type: Boolean

        IMAP check options:

            * port -- Target server port
                    Type: Integer

            * stringtoexpect -- String to expect in response
                    Type: String

            * encryption -- Use connection encryption
                    Type: Boolean
        """

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['paused', 'resolution', 'contactids', 'sendtoemail',
                           'sendtosms', 'sendtotwitter', 'sendtoiphone',
                           'sendnotificationwhendown', 'notifyagainevery',
                           'notifywhenbackup', 'created', 'type', 'hostname',
                           'status', 'lasterrortime', 'lasttesttime', 'url',
                           'encryption', 'port', 'auth', 'shouldcontain',
                           'shouldnotcontain', 'postdata', 'additionalurls',
                           'stringtosend', 'stringtoexpect', 'expectedip',
                           'nameserver', 'use_legacy_notifications', 'host',
                           'alert_policy', 'autoresolve', 'probe_filters']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument of' +
                                 '<PingdomCheck>.modify()\n')

        # If one of the legacy parameters is used, it is required to set the legacy flag.
        # https://github.com/KennethWilke/PingdomLib/issues/12
        if any([k for k in kwargs if k in legacy_notification_parameters]):
            if "use_legacy_notifications" in kwargs and kwargs["use_legacy_notifications"] != True:
                raise Exception("Cannot set legacy parameter when use_legacy_notifications is not True")
            kwargs["use_legacy_notifications"] = True

        response = self.pingdom.request("PUT", 'checks/%s' % self.id, kwargs)

        return response.json()['message']