from singleton import Singleton
import sys
from time import localtime, strftime, time


class TermColors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LogLevel(object):
    WARN = 5
    INFO = 6
    DEBUG = 7
    LOG_LEVEL_MAP = {
        "debug": DEBUG,
        "warn":  WARN,
        "info":  INFO,
    }


class Logger(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.log_level = LogLevel.INFO
        self.ln_len = 100

    def set_log_level(self, log_level):
        if type(log_level) == int:
            self.log_level = log_level
        elif type(log_level) == str:
            try:
                self.log_level = LogLevel.LOG_LEVEL_MAP[log_level.lower()]
            except KeyError:
                self.logw("Unknown log level")

    def _print(self, text, *args, **kwargs):
        if not type(text) == str:
            text = str(text)
        sys.stdout.write("\r" + " " * self.ln_len)
        sys.stdout.write(("\r[{}] " + text).format(strftime("%m-%d %H:%M:%S",
                                                            localtime()),
                                                   *args)[:self.ln_len])
        sys.stdout.flush()

    def log(self, text, *args, **kwargs):
        if self.log_level < LogLevel.INFO:
            return
        self._print(text, *args, **kwargs)

    def logw(self, text, *args):
        if self.log_level >= LogLevel.WARN:
            self._print("{}[WARN] {}{}\n".format(TermColors.WARN,
                                                 TermColors.ENDC,
                                                 text),
                        *args, force=True)

    def logd(self, text, *args):
        if self.log_level >= LogLevel.DEBUG:
            self._print("{}[DEBUG] {}{}\n".format(TermColors.OKBLUE,
                                                  TermColors.ENDC,
                                                  text),
                        *args)


logger = Logger()
set_log_level = logger.set_log_level
log = logger.log
logd = logger.logd
logw = logger.logw


class TimeScope(object):
    def __init__(self, name, debug_only=False):
        self.name = name
        self.debug_only = debug_only

    def __enter__(self):
        self.time = time()

    def __exit__(self, *args):
        if self.debug_only:
            logd("{} computation finished in {:.3f} secs",
                 self.name, time() - self.time)
        else:
            log("{} computation finished in {:.3f} secs\n",
                self.name, time() - self.time)
