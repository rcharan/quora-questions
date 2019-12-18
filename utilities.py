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
# Part 1b : Some not so simple convenience functions
#   See http://hackage.haskell.org/package/base-4.12.0.0/docs/Control-Applicative.html#v:liftA2
#    for inspiration.
#
#   This is restricted to the applicative functor f = (-> r) aka functions
#
################################################################################

# liftA :: (a -> b) -> f a -> f b
#  This is post-composition

def no_predicates(*funcs):
    def f(arg):
        return not(any(func(arg) for func in funcs))
    return f
################################################################################
#
# Part 2 : Some simple convenience functions for dropping data in dataframes
#
################################################################################

def drop_col(df, *cols):
    df.drop(columns = list(cols), inplace = True)


def drop_by_rule(df, bool_series):
    index = df[bool_series].index
    df.drop(index = index, inplace = True)

################################################################################
#
# Part 3 : Convenience functions for plotting
#
################################################################################

# With pyplot interactive mode off (via plt.ioff() call)
#  every call needs a figure and axis created
#  this provides a simple way to create such a figure
def plot(fn, *args, **kwargs):
    if 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig, ax = plt.subplots()
    kwargs['ax'] = ax
    fn(*args, **kwargs)
    return fig


################################################################################
#
# Part 4 : Timing
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
# Part 5 : Persistence and timing conveniences
#
################################################################################

fit_time_fname = './models/fit_times.joblib'

def _fit_time_interface(model_name, write = None):
    if os.path.exists(fit_time_fname):
        fit_time_dict = load(fit_time_fname)
    else:
        if not write:
            print('No fit time info found')
            return None
        fit_time_dict = {}

    if write:
        fit_time_dict[model_name] = write
        dump(fit_time_dict, fit_time_fname)
    else:
        if model_name in fit_time_dict:
            return fit_time_dict[model_name]
        else:
            print(f'No fit time found for {model_name}')
            return None

def get_fit_time(model_name):
    return _fit_time_interface(model_name)

def write_fit_time(model_name, fit_time):
    return _fit_time_interface(model_name, fit_time)

def get_time_per_fold(model_data, param_grid, splitter = None):
    fit_time = float(model_data['fit_time'])
    n_params = functools.reduce(lambda x, y : x * y,
        (len(v) for v in param_grid.values())
    )
    if splitter:
        try:
            n_folds = splitter.get_n_splits() * splitter.n_repeats
        except AttributeError:
            n_folds = splitter.get_n_splits()
    else:
        n_folds = 1

    return round(fit_time / (n_params * n_folds), 2)



# def persist_fit_transform(model_name,
#                           model = None,
#                           data  = None,
#                           logic = 'fit'
#                           force_refit = False):
#
#     '''
#         Allows two logics: logic = 'fit' or 'fit_transform'
#
#         If 'fit', then one of (in order of priority)
#             (a) load model from disk; or
#             (b) fit it with the data
#           and return the model and model_data
#         If 'fit_transform', then one of (in order of priority):
#             (a) load model and data from disk
#             (b) load model and transform the given data
#             (c) fit_transform the model
#           and return the model and transformed data and model_data
#
#         Files are always in './models/' directory
#
#         fit or fit_transform times are always reported
#
#         return is model, transformed data (if fit_transform only), model_data dict
#
#         force_fit causes both fit and (if fit_transform is set) transform
#
#         Data: either X or [X, y]
#
#     '''
#     if data and not isinstance(data, list_):
#         data = [data]
#
#     assert logic in ['fit', 'fit_transform']
#
#     paths = {
#         'model' : f'./models/{model_name}-model.joblib',
#         'data'  : f'./models/{model_name}-data.joblib'
#     }
#
#     path_exists = {k, os.path.exists(v) for k, v in paths.items()}
#
#     load_ = {}
#     load_['model'] = (not force_fit) and path_exists['model']
#     load_['data']  = (not force_fit) and path_exists['data']  and \
#                       logic == 'fit_transform' and load_['model']
#
#     model_data = {}
#
#     outputs = {}
#     for k in load_:
#         if not load_[k]:
#             continue
#
#         outputs[k] = load(paths[k])
#         print(f'Loading {k} from disk')
#
#     if any(load_.values())
#         fit_time = get_fit_time(model_name)
#         model_data['fit_time'] = fit_time
#         if fite_time:
#             print(f'''{fit_time} seconds elapsed in original {logic}''')
#         else:
#             print('No fit time was found')
#
#     if not load_['model']
#         if logic == 'fit':
#             print(f'Fitting model {model_name}')
#             Timer.start()
#             model.fit(*data)
#             Timer.end()
#             dump(model, f'./models/{model_name}.joblib')
#         else:
#             print(f'Fitting model and transforming data')
#             Timer.start()
#             out_data = model.fit_transform(*data)
#             Timer.end()
#             dump(model, paths['model'])
#             dump(out_data, paths['data'])
#             model_data['transformed data']  = out_data
#             outputs['data'] = out_data
#
#
#         outputs['model'] = model
#         model_data['fit_time']          = Timer.time
#         model_data['model']             = model
#         write_fit_time(model_name, model_data['fit_time'])
#
#     elif load_['data']:
#         print(f'Transforming data')
#         Timer.start()
#         out_data = model.transform()
#
#
#
#
#
#     if not load_from_disk or model_not_found or force_fit:
#
#         # Write information
#
#     models[model_name] = model_data
#
#     return model, model_data

def fit_or_load_model(model, data, model_name, force_fit = False):
    model_not_found = False
    model_data = {'model' : model}
    if not force_fit:
        try:
            model = load(f'./models/{model_name}.joblib')
        except:
            model_not_found = True
        else:
            print(f'Loading model from disk')
            fit_time = get_fit_time(model_name)
            model_data['fit_time'] = fit_time
            print(f'''{fit_time} seconds elapsed in original fitting''')

    if model_not_found or force_fit:
        print(f'Fitting model {model_name}')

        # Fit the Model
        Timer.start()
        model.fit(data)
        Timer.end()

        # Write information
        model_data['fit_time'] = Timer.time
        dump(model, f'./models/{model_name}.joblib')
        write_fit_time(model_name, model_data['fit_time'])

    return model, model_data



################################################################################
#
# Part 6 : ROC Curve
#
################################################################################

def plot_roc_curve(fpr, tpr):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label = 'ROC')
    xs = np.linspace(0, 1, len(fpr))
    ax.plot(xs, xs, label = 'Diagonal')
    ax.set_xlim([-0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.grid(True)
    ax.set_aspect(1)
    ax.legend()
    return fig
