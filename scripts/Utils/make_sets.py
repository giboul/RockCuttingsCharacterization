from os import listdir, mkdir
from os.path import join, isdir, dirname
from pandas import DataFrame
from Utils import path_to
from pandas import read_csv


"""This script is a place for preparing the datasets (*.csv)
 in the /config/sets folder.
No other script depends on this one"""


def create_dataset(to_folder, data_folders=None, keyword=""):
    """Creates a dataset with all files
    Args:
        data_folders: Iterable=None: only looks into the given folders
        keyword: str="": filters folders to only thow whose name
         contains the keyword
    """

    set_name = f"{'-'.join(data_folders)}.csv"

    # Corresponding labels
    label_dict = dict(
        BL=0,
        GN=1,
        ML=2,
        MS=3,
        OL=4  # Add RV=5 when there is enough data
    )

    data_folder = 'data'

    folder_path = path_to('config', 'sets', dirname(set_name))
    if not isdir(folder_path):
        print(f"Creating {folder_path}")
        mkdir(folder_path)

    # print space lengths, just for the show in the output
    _max_len = max([len(d) for d in listdir(path_to(data_folder))])
    _max_len1 = max([
        max([
            len(f) for f in listdir(path_to(data_folder, d))
        ])
        for d in listdir(path_to(data_folder))
    ])

    print(f"{'Set':^{_max_len}} : {'Folder':^{_max_len1}} |"
          f" {'Rock':^4} = {'label':^5}")
    paths = []
    # Iterate over all dataset's files
    if not data_folders:
        data_folders = listdir(path_to(data_folder))
    for data_set in data_folders:
        if not data_set == 'Raw':
            folders = listdir(path_to(data_folder, data_set))
            # Filter the folders by keyword (was useful for giboul)
            folders = [folder for folder in folders if keyword in folder]
            for folder in folders:
                # Rock type acronym should be the first two letters
                # of the folder name
                rock = folder[:2]
                if rock in label_dict:
                    label = label_dict[rock]
                    print(
                        f"{data_set:^{_max_len}} : {folder:^{_max_len1}} | "
                        f"{rock:^4} = {label:^5}")
                    for file in listdir(path_to(
                        data_folder, data_set, folder
                    )):
                        # add file to file list here
                        # if you need to filter it, here may be the best place
                        paths.append([
                            join(data_folder, data_set, folder, file),
                            folder,
                            label,
                        ])

    # Save to *.csv file
    df = DataFrame(paths, columns=('Paths', 'Folders', 'Label'))
    to_folder_path = path_to('config', 'sets', to_folder)
    if not isdir(to_folder_path):
        mkdir(to_folder_path)
    df.to_csv(path_to('config', 'sets', to_folder, set_name))


def modify():
    """If you need to replace, addapt or renew paths in *.csv files
    Overwrite doesn't happen unless you uncomment the df.to_csv(...) line"""

    # This shouldn't change
    data_folder = 'config/sets'

    # Iterating over all folders
    for dir in [
        dir for dir in listdir(path_to(data_folder))
        if isdir(path_to(data_folder, dir))
    ]:
        for file in listdir(path_to(data_folder, dir)):
            df = read_csv(path_to(data_folder, dir, file))
            # Count how many values you'll change
            freq = len(df[df.Paths.str.contains('/debug/', regex=False)])
            print(f"{dir:^8} : {file:^20} -> {freq = }")
            # Replace values
            df.Paths = df.Paths.str.replace('/debug/', '/New/', regex=False)
            # Some extra clumns were neing added...
            columns = [
                column for column in df.columns if 'Unnamed' not in column]
            df = df[columns]
            print(f"Saving as {path_to(data_folder, dir, file)}")
            df.to_csv(path_to(data_folder, dir, file))


if __name__ == '__main__':
    print("Running make_sets.py...")
    # create_dataset(data_folders=('Borehole', ), to_folder='all')
    # create_dataset(
    #   path_to('config', 'sets', 'debug', 'debug.csv'),
    #   keyword='debug'
    # )
    # modify()
