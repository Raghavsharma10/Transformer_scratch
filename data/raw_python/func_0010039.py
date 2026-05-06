def __unregister_driver(self, channel):
        "Unregister webdriver"
        driver = self.__get_driver_for_channel(channel)

        if self.__registered_drivers.has_key(channel) \
                and driver in self.__registered_drivers[channel]:

            self.__registered_drivers[channel].remove(driver)

        self.__webdriver[channel] = None