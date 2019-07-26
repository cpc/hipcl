ENV variables controlling behaviour
---------------------------------------

The behavior of HIPCL can be controlled with multiple environment variables
listed below. The variables are helpful both when using and when developing
pocl.

- **HIPCL_LOGLEVEL**
  Changes verbosity of log messages coming from HIPCL.
  Possible values are: debug,info,warn,err,crit,off
  Defaults to "err". HIPCL will log messages of this priority and higher.

