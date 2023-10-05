import argparse

parser = argparse.ArgumentParser(description='Add paths for Dolphin and Melee ISO')

parser.add_argument('--dolphin_executable_path', '-e', default=None,
                    help='The path to Slippi/Dolphin app')

parser.add_argument('--iso', default=None, type=str,
                    help='Path to melee iso')

def get_paths():
    parser = argparse.ArgumentParser(description='Example of libmelee in action')

    parser.add_argument('--dolphin_exe_path', '-d', default=None,
                    help='The path to Slippi/Dolphin app')

    parser.add_argument('--melee_iso', '-i', default=None, type=str,
                    help='Path to melee iso')
    
    args = parser.parse_args()

    return args.dolphin_exe_path, args.melee_iso