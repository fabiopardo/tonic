import argparse
import os
import time

import wandb
import yaml
import csv


class WandbProcessor:
    """
    Class that takes a log.csv file from tonic and 
    logs it with wandb. It periodically checks if the
    log.csv has been updated and pushes the new changes.
    """

    def __init__(self, path):
        self._path = path
        self._current_line_number = 0
        self._last_line_number = 0
        self._last_update = os.path.getmtime(path)
        self._setup_wandb()

    def update(self):
        self._last_update, updated = check_if_csv_has_updated(
            self._path, self._last_update
        )
        if updated:
            data = load_csv_to_dict(self._path)
            self._current_line_number = self._get_line_number(data)
            self._log(data)
            self._last_line_number = self._get_line_number(data)

    def _get_line_number(self, data):
        return len(data["train/episode_score/mean"])

    def _setup_wandb(self):
        data = load_csv_to_dict(self._path)
        self._current_line_number = self._get_line_number(data)
        self._log(data)
        self._last_line_number = self._get_line_number(data)

    def _log(self, data):
        """
        Log the data, but remove the mean keyword from the 
        logged name such that you can see all train/ and test/
        data in one section.
        """
        for idx in range(self._last_line_number, self._current_line_number):

            logged_data = {k: v[idx] for k, v in data.items()}
            change = 1
            while change:
                for k, v in logged_data.items():
                    if '/mean' in k:
                        logged_data.pop(k)
                        logged_data[k[:-5]] = v
                        change = 1
                        break
                    change = 0
            wandb.log(logged_data, step=idx)


def check_if_csv_has_updated(csv_path, last_update):
    """
    Checks if the csv file has been updated since the last write.
    """
    current_update = os.path.getmtime(csv_path)
    if current_update > last_update:
        return current_update, True
    return current_update, False


def load_csv(csv_path):
    """
    Loads the csv file and returns the data.
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def load_csv_to_dict(csv_path):
    """
    Converts the csv file to a dictionary.
    """
    data = load_csv(csv_path)
    keys = data[0]
    data = data[1:]
    data = [list(map(float, x)) for x in data if x[0] != "None"]
    data = {k: [x[idx] for x in data] for idx, k in enumerate(keys)}
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()
    config = yaml.load(
        open(os.path.join(args.path[:-7], "config.yaml"), "r"), Loader=yaml.FullLoader
    )
    wandb.init(project=args.project, entity=args.entity, config=config)
    processor = WandbProcessor(args.path)
    while True:
        processor.update()
        time.sleep(100)
