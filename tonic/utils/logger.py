import copy
import datetime
import os
import time

import numpy as np
import termcolor
import yaml


current_logger = None


class Logger:
    '''Logger used to display and save logs, and save experiment configs.'''

    def __init__(self, path=None, width=60, script_path=None, config=None):
        self.path = path or str(time.time())

        # Save the launch script.
        if script_path:
            with open(script_path, 'r') as script_file:
                script = script_file.read()
                try:
                    os.makedirs(self.path, exist_ok=True)
                except Exception:
                    pass
                script_path = os.path.join(self.path, 'script.py')
                with open(script_path, 'w') as config_file:
                    config_file.write(script)
                log('Script file saved to {}'.format(script_path))

        # Save the configuration.
        if config:
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass
            config_path = os.path.join(self.path, 'config.yaml')
            with open(config_path, 'w') as config_file:
                yaml.dump(config, config_file)
            log('Config file saved to {}'.format(config_path))

        self.first_row = True
        self.stat_keys = set()
        self.epoch_dict = {}
        self.width = width
        self.last_epoch_progress = None
        self.start_time = time.time()

    def store(self, key, value, stats=False):
        '''Keeps named values during an epoch.'''

        if self.first_row and stats:
            self.stat_keys.add(key)

        if key not in self.epoch_dict.keys():
            self.epoch_dict[key] = [copy.copy(value)]
        else:
            self.epoch_dict[key].append(copy.copy(value))

    def dump(self):
        '''Displays and saves the values at the end of an epoch.'''

        # Compute statistics if needed.
        keys = list(self.epoch_dict.keys())
        for key in keys:
            if key in self.stat_keys:
                self.epoch_dict[key + '/mean'] = np.mean(self.epoch_dict[key])
                self.epoch_dict[key + '/std'] = np.std(self.epoch_dict[key])
                self.epoch_dict[key + '/min'] = np.min(self.epoch_dict[key])
                self.epoch_dict[key + '/max'] = np.max(self.epoch_dict[key])
                self.epoch_dict[key + '/size'] = len(self.epoch_dict[key])
                del self.epoch_dict[key]
            else:
                self.epoch_dict[key] = np.mean(self.epoch_dict[key])

        # List the keys and prepare the display layout.
        if self.first_row:
            self.final_keys = list(sorted(self.epoch_dict.keys()))
            self.console_formats = []
            known_keys = set()
            for key in self.final_keys:
                *left_keys, right_key = key.split('/')
                for i, k in enumerate(left_keys):
                    left_key = '/'.join(left_keys[:i + 1])
                    if left_key not in known_keys:
                        left = '  ' * i + k.replace('_', ' ')
                        self.console_formats.append((left, None))
                        known_keys.add(left_key)
                indent = '  ' * len(left_keys)
                right_key = right_key.replace('_', ' ')
                self.console_formats.append((indent + right_key, key))

        # Display the values following the layout.
        print()
        for left, key in self.console_formats:
            if key:
                val = self.epoch_dict.get(key)
                str_type = str(type(val))
                if 'tensorflow' in str_type:
                    warning('Logging TensorFlow tensor {}'.format(key))
                elif 'torch' in str_type:
                    warning('Logging Torch tensor {}'.format(key))
                if np.issubdtype(type(val), np.floating):
                    right = '{:8.3g}'.format(val)
                elif np.issubdtype(type(val), np.integer):
                    right = '{:,}'.format(val)
                else:
                    right = str(val)
                spaces = ' ' * (self.width - len(left) - len(right))
                print(left + spaces + right)
            else:
                spaces = ' ' * (self.width - len(left))
                print(left + spaces)
        print()

        # Save the data to the log file
        log_file_path = os.path.join(self.path, 'log.csv')
        if self.first_row:
            vals = [self.epoch_dict[key] for key in self.final_keys]
            log('Logging data to {}'.format(log_file_path))
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass
            with open(log_file_path, 'w') as file:
                file.write(','.join(self.final_keys) + '\n')
                file.write(','.join(map(str, vals)) + '\n')
        else:
            vals = [self.epoch_dict.get(key) for key in self.final_keys]
            with open(log_file_path, 'a') as file:
                file.write(','.join(map(str, vals)) + '\n')

        self.epoch_dict.clear()
        self.first_row = False
        self.last_epoch_progress = None
        self.last_epoch_time = time.time()

    def show_progress(
        self, steps, num_epoch_steps, num_steps, color='white',
        on_color='on_blue'
    ):
        '''Shows a progress bar for the current epoch and total training.'''

        epoch_steps = (steps - 1) % num_epoch_steps + 1
        epoch_progress = int(self.width * epoch_steps / num_epoch_steps)
        if epoch_progress != self.last_epoch_progress:
            current_time = time.time()
            seconds = current_time - self.start_time
            seconds_per_step = seconds / steps
            epoch_rem_steps = num_epoch_steps - epoch_steps
            epoch_rem_secs = max(epoch_rem_steps * seconds_per_step, 0)
            epoch_rem_secs = datetime.timedelta(seconds=epoch_rem_secs + 1e-6)
            epoch_rem_secs = str(epoch_rem_secs)[:-7]
            total_rem_steps = num_steps - steps
            total_rem_secs = max(total_rem_steps * seconds_per_step, 0)
            total_rem_secs = datetime.timedelta(seconds=total_rem_secs)
            total_rem_secs = str(total_rem_secs)[:-7]
            msg = 'Time left:  epoch {}  total {}'.format(
                epoch_rem_secs, total_rem_secs)
            msg = msg.center(self.width)
            print(termcolor.colored(
                '\r' + msg[:epoch_progress], color, on_color), end='')
            print(msg[epoch_progress:], sep='', end='')
            self.last_epoch_progress = epoch_progress


def initialize(*args, **kwargs):
    global current_logger
    current_logger = Logger(*args, **kwargs)
    return current_logger


def get_current_logger():
    global current_logger
    if current_logger is None:
        current_logger = Logger()
    return current_logger


def store(*args, **kwargs):
    logger = get_current_logger()
    return logger.store(*args, **kwargs)


def dump(*args, **kwargs):
    logger = get_current_logger()
    return logger.dump(*args, **kwargs)


def show_progress(*args, **kwargs):
    logger = get_current_logger()
    return logger.show_progress(*args, **kwargs)


def get_path():
    logger = get_current_logger()
    return logger.path


def log(msg, color='green'):
    print(termcolor.colored(msg, color, attrs=['bold']))


def warning(msg, color='yellow'):
    print(termcolor.colored('Warning: ' + msg, color, attrs=['bold']))


def error(msg, color='red'):
    print(termcolor.colored('Error: ' + msg, color, attrs=['bold']))
