import os
from glob import glob


def read_dirfile_name(data_dir, data_type='jpg'):
  data_type = '*' + data_type
  all_files = glob(os.path.join(data_dir, data_type))
  all_files = [file.split('.')[0] for file in all_files]
  return all_files


def read_dirfile_name_type(data_dir, data_type='jpg'):
  data_type = '*' + data_type
  all_files = glob(os.path.join(data_dir, data_type))
  return all_files


def license_plate(all_files):
  name_labels = [file.split('/')[-1].split('_')[0] for file in all_files]
  return name_labels

if __name__ == '__main__':
  DATA_DIR = '/home/woodylin/tensorflow3/Plate_Recognition/data/train/'
  all_files = read_dirfile_name(DATA_DIR)
  labels = license_plate(all_files)
  print(1)