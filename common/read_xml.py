import os
import xml.etree.ElementTree as ET


def read_location(file_path):
  tree = ET.ElementTree(file=file_path)
  root = tree.getroot()
  for bndbox in root.iter('bndbox'):
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
    return xmin, ymin, xmax, ymax


def read_text():
  return 1


if __name__ == '__main__':
  path = os.path.join(os.path.dirname(__file__), 'test.xml')
  print(read_location(path))
