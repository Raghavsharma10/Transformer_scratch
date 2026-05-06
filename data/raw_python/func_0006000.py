def is_alert_present(self):
        """Tests if an alert is present

        @return: True if alert is present, False otherwise
        """
        current_frame = None
        try:
            current_frame = self.driver.current_window_handle
            a = self.driver.switch_to_alert()
            a.text
        except NoAlertPresentException:
            # No alert
            return False
        except UnexpectedAlertPresentException:
            # Alert exists
            return True
        finally:
            if current_frame:
                self.driver.switch_to_window(current_frame)
        return True