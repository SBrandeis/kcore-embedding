import time


def timeit(var_name):
    def wrapper(func):
        def timed_f(self, *args, **kwargs):
            start = time.time()
            res = func(self, *args, **kwargs)
            end = time.time()
            self.times[var_name] = end - start
            return res
        return timed_f
    return wrapper
