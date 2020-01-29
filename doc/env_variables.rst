ENV variables controlling behaviour
---------------------------------------

The behavior of HIPCL can be controlled with multiple environment variables
listed below. The variables are helpful both when using and when developing
pocl.

- **HIPCL_LOGLEVEL**
  String value. Changes verbosity of log messages coming from HIPCL.
  Possible values are: debug,info,warn,err,crit,off
  Defaults to "err". HIPCL will log messages of this priority and higher.

- **HIPCL_PLATFORM**
  Numeric value. If there are multiple OpenCL platforms on the system, setting this to a number (0..platforms-1)
  will limit HipCL to that single platform. By default HipCL can access all OpenCL platforms.

- **HIPCL_DEVICE**
  Numeric value. If there are multiple OpenCL devices in the selected platform, setting this to a number (0..N-1)
  will limit HipCL to a single device. If HIPCL_PLATFORM is not set but HIPCL_DEVICE is,
  HIPCL_PLATFORM defaults to 0.

- **HIPCL_DEVICE_TYPE**
  String value. Limits OpenCL device visibility to HipCL based on device type.
  Possible values are: all, cpu, gpu, default, accel

