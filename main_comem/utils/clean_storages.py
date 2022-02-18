import argparse
import shutil
import os
import pickle

from alfred.utils.config import parse_bool
from alfred.utils.misc import select_storage_dirs
from alfred.utils.directory_tree import DirectoryTree

from main_comem.evaluate import get_eval_args, evaluate


def get_analyse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--storage_names', type=str, nargs='+', default=None)

    parser.add_argument('--folder_to_remove', type=str)
    parser.add_argument('--file_to_keep', type=str)
    parser.add_argument('--remove_anyway', type=parse_bool)

    parser.add_argument('--root_dir', default=None, type=str)

    return parser.parse_args()

def _clean_storages(args):
    storage_dirs = select_storage_dirs(args.from_file, args.storage_names, args.root_dir)
    for storage_dir in storage_dirs:
        experiment_dirs = DirectoryTree.get_all_experiments(storage_dir)
        for experiment_dir in experiment_dirs:
            seed_dirs = DirectoryTree.get_all_seeds(experiment_dir)
            for seed_dir in seed_dirs:
                if (seed_dir / args.folder_to_remove).exists():
                    if args.remove_anyway:
                        print(f'Removing {seed_dir / args.folder_to_remove}')
                        shutil.rmtree(seed_dir / args.folder_to_remove)
                    else:
                        if (seed_dir / args.folder_to_remove/args.file_to_keep).exists():
                            continue
                        else:
                            subdirs = [x for x in (seed_dir / args.folder_to_remove).iterdir() if x.is_dir()]
                            kept = 0
                            for subdir in subdirs:
                                if not (subdir/args.file_to_keep).exists():
                                    print(f'Removing {subdir}')
                                    shutil.rmtree(subdir)
                                else:
                                    kept += 1
                            if kept == 0:
                                print(f'Removing {seed_dir / args.folder_to_remove}')
                                shutil.rmtree(seed_dir / args.folder_to_remove)


if __name__ == '__main__':
    args = get_analyse_args()
    _clean_storages(args)