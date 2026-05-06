def delete_alert(self, email, alert_id):
        """
        delete user alert
        """
        data = {'email': email,
                'alert_id': alert_id}
        return self.api_delete('alert', data)