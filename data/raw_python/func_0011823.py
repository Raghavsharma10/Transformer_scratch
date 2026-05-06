def send_results(self):
        '''
        send results
        '''

        for server in self.servers:
            if self.servers[server]['results']:
                if len(self.servers[server]['results']) == 1:
                    msg = MIMEText('')
                    msg['Subject'] = '[%(custom_fqdn)s] [%(service_description)s] %(return_status)s: %(output)s' % self.servers[server]['results'][0]
                else:
                    txt = ''
                    summary = [0, 0, 0, 0]
                    for results in self.servers[server]['results']:
                        txt += '[%(service_description)s] %(return_status)s: %(output)s\n' % results
                        summary[results['return_code']] += 1
                    msg = MIMEText(txt)
                    subject = '[%(custom_fqdn)s]' % self.servers[server]['results'][0]
                    for i, status in enumerate(STATUSES):
                        subject += ' %s:%s' % (status[0], summary[i])
                    msg['Subject'] = subject

                msg['From'] = self.servers[server]['from']
                msg['To'] = ', '.join(self.servers[server]['to'])
                if self.servers[server]['tls']:
                    smtp_server = smtplib.SMTP_SSL(self.servers[server]['host'], self.servers[server]['port'])
                else:
                    smtp_server = smtplib.SMTP(self.servers[server]['host'], self.servers[server]['port'])

                if self.servers[server]['login'] and len(self.servers[server]['login']) > 0:
                    smtp_server.login(self.servers[server]['login'], self.servers[server]['password'])
                smtp_server.sendmail(self.servers[server]['from'], self.servers[server]['to'], msg.as_string())
                smtp_server.quit()
                LOG.info("[email][%s]: e-mail sent from: %s to: %s", server, self.servers[server]['from'], self.servers[server]['to'])