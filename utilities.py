import datetime
import matplotlib.pyplot as plt
import functools
import numpy as np
import os
from joblib import dump, load


################################################################################
#
# Part 1 : Some simple convenience functions
#
################################################################################

def compose(*funcs):
    outer = funcs[:-1][::-1] # Python only provides left folds
    def composed_func(*args, **kwargs):
        inner = funcs[-1](*args, **kwargs)
        return functools.reduce(lambda val, func : func(val), outer, inner)
    return composed_func

# Aliases for filters and maps
lfilter        = compose(list, filter)
lmap           = compose(list, map)
afilter        = compose(np.array, list, filter)
filternull     = functools.partial(filter, None)
lfilternull    = compose(list, filternull)
filternullmap  = compose(filternull, map)
lfilternullmap = compose(lfilternull, map)
def lmap_filter(map_, predicate_):
    def f(list_):
        return lmap(map_, filter(predicate_, list_))
    return f

def list_diff(a, b):
    return list(set(a).difference(b))

def print_dict(d):
    key_len = max(map(len, d.keys()))
    for k, v in d.items():
        print(f'{k.ljust(key_len)} : {v}')



################################################################################
#
# Part 2 : Timing
#
################################################################################

# Super simple timer
#  Timing implemented as class methods
#  to avoid having to instantiate
class Timer:

    @classmethod
    def start(cls):
        cls.start_time = datetime.datetime.now()

    @classmethod
    def end(cls):
        delta     = datetime.datetime.now() - cls.start_time
        sec       = delta.seconds
        ms        = delta.microseconds // 1000
        cls.time  = f'{sec}.{ms}'
        print(f'{sec}.{ms} seconds elapsed')


################################################################################
#
# Part 3 : Persistence and timing conveniences
#
################################################################################

def persist(obj_name, model, method, *data, task = 'model', force_fit = False, **kwargs):
    '''
        Persist a call (e.g. fit, fit_transform, score)

        Attempts to load from disk, otherwise makes the call and saves it

        task: either model, data, or both
    '''
    file_name = f'./models/{obj_name}.joblib'

    if os.path.exists(file_name) and not force_fit:
        print(f'Loading {obj_name} from disk')
        obj      = load(file_name)
        job_time = get_fit_time(obj_name)
        return obj

    else:
        assert task in ['model', 'data', 'both']
        print(f'Running {method}')
        Timer.start()
        return_val = model.__getattribute__(method)(*data, **kwargs)
        Timer.end()
        if task == 'model':
            out = model
        elif task == 'data':
            out = return_val
        elif task == 'both':
            out = (return_val, model)

        dump(out, file_name)
        write_fit_time(obj_name, Timer.time)
        return out


fit_time_fname = './models/fit_times.joblib'

def _fit_time_interface(model_name, write = None):
    if os.path.exists(fit_time_fname):
        fit_time_dict = load(fit_time_fname)
    else:
        if not write:
            print('No job time info found')
            return None
        fit_time_dict = {}

    if write:
        fit_time_dict[model_name] = write
        dump(fit_time_dict, fit_time_fname)
    else:
        if model_name in fit_time_dict:
            job_time = fit_time_dict[model_name]
            print(f'{job_time} seconds elapsed in original job')
            return job_time
        else:
            print(f'No job time found for {model_name}')
            return None

def get_fit_time(model_name):
    return _fit_time_interface(model_name)

def write_fit_time(model_name, fit_time):
    return _fit_time_interface(model_name, fit_time)
