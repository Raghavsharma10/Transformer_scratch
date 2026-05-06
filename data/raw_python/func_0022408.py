def set_serial(self, android_serial):
        """
        Specify given *android_serial* device to perform test.

        You do not have to specify the device when there is only one device connects to the computer.

        When you need to use multiple devices, do not use this keyword to switch between devices in test execution.

        Using different library name when importing this library according to http://robotframework.googlecode.com/hg/doc/userguide/RobotFrameworkUserGuide.html?r=2.8.5.

        | Setting | Value  | Value     | Value   | 
        | Library | Mobile | WITH NAME | Mobile1 |
        | Library | Mobile | WITH NAME | Mobile2 |

        And set the serial to each library.

        | Test Case        | Action             | Argument           |
        | Multiple Devices | Mobile1.Set Serial | device_1's serial  |
        |                  | Mobile2.Set Serial | device_2's serial  |

        """
        self.adb = ADB(android_serial)
        self.device = Device(android_serial)
        self.test_helper = TestHelper(self.adb)