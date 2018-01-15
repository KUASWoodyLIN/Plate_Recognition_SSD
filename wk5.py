import os
import re
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd

def load_xml(file_path):
  tree = ET.ElementTree(file=file_path)
  root = tree.getroot()
  folder = root[0].text
  filename = root[1].text
  size = [int(root[4][0].text), int(root[4][1].text), int(root[4][2].text)]
  object_name = root[6][0].text
  xmin = int(root[6][4][0].text)
  ymin = int(root[6][4][1].text)
  xmax = int(root[6][4][2].text)
  ymax = int(root[6][4][3].text)
  return folder, filename, size, object_name, xmin, ymin, xmax, ymax


data_dir = os.getcwd()
pattern = re.compile('(.+\/)?(\w+)\/([^_]+)_.+xml')
all_files = glob(os.path.join(data_dir + '/data/train/*xml'))
all_files = [re.sub(r'\\', r'/', file) for file in all_files]
print('Number of training data:', len(all_files))
print()


frames = []
xmins = []
xmaxs = []
ymins = []
ymaxs = []
ids = ['background']
class_ids = []

for entry in all_files:
  r = re.match(pattern, entry)
  if r:
    folder, filename, size, object_name, xmin, ymin, xmax, ymax = load_xml(entry)
    file_name, file_extension = os.path.splitext(filename)
    if not file_name in entry:
      print(filename, entry)
      continue
    frames.append(filename)
    id = filename.split('_')[0]
    if id not in ids:
      ids.append(id)
    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)
    class_ids.append(ids.index(id))

frames = [re.sub(r'xml', r'jpg', frame) for frame in frames]
train_labels = pd.DataFrame({'frame': frames, 'xmin': xmins, 'xmax': xmaxs,
                             'ymin': ymins, 'ymax': ymaxs, 'class_id': class_ids})
train_labels = train_labels[['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']]
train_labels.to_csv('train_labels.csv', index=False)

val_labels = pd.DataFrame({'frame': frames, 'xmin': xmins, 'xmax': xmaxs,
                           'ymin': ymins, 'ymax': ymaxs, 'class_id': class_ids})
val_labels = val_labels[['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']]
val_labels.to_csv('val_labels.csv', index=False)


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from Plate_Recognition_SSD.keras_ssd7 import build_model
from Plate_Recognition_SSD.keras_ssd_loss import SSDLoss
from Plate_Recognition_SSD.keras_layer_AnchorBoxes import AnchorBoxes
from Plate_Recognition_SSD.keras_layer_L2Normalization import L2Normalization
from Plate_Recognition_SSD.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from Plate_Recognition_SSD.ssd_batch_generator import BatchGenerator

img_height = 240
img_width = 320
img_channels = 3
n_classes = len(ids)
min_scale = 0.08
max_scale = 0.96
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
aspect_ratios = [0.5, 1.0, 2.0]
two_boxes_for_ar1 = True
limit_boxes = False
variances = [1.0, 1.0, 1.0, 1.0]
coords = 'centroids'
normalize_coords = False

K.clear_session()
model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=min_scale,
                                     max_scale=max_scale,
                                     scales=scales,
                                     aspect_ratios_global=aspect_ratios,
                                     aspect_ratios_per_layer=None,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

train_images_path = '.data/train'
train_labels_path = './train_labels.csv'
val_images_path = '.data/train'
val_labels_path = './val_labels.csv'
train_dataset.parse_csv(images_path=train_images_path,
                        labels_path=train_labels_path,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_path=train_images_path,
                      labels_path=train_labels_path,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                scales=scales,
                                aspect_ratios_global=aspect_ratios,
                                aspect_ratios_per_layer=None,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

batch_size = 16
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5),
                                         flip=0.5,
                                         translate=((5, 50), (3, 30), 0.5),
                                         scale=(0.75, 1.3, 0.5),
                                         max_crop_and_resize=False,
                                         full_crop_and_resize=False,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=(0.5, 2, 0.5),
                                     flip=0.5,
                                     translate=((5, 50), (3, 30), 0.5),
                                     scale=(0.75, 1.3, 0.5),
                                     max_crop_and_resize=False,
                                     full_crop_and_resize=False,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4)

n_train_samples = train_dataset.get_n_samples()
n_val_samples = val_dataset.get_n_samples()

epochs = 10

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(n_train_samples/batch_size),
                              epochs=epochs,
                              callbacks=[ModelCheckpoint('ssd7_weights_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         mode='auto',
                                                         period=1),
                                         EarlyStopping(monitor='val_loss',
                                                       min_delta=0.001,
                                                       patience=2),
                                         ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.5,
                                                           patience=0,
                                                           epsilon=0.001,
                                                           cooldown=0)],
                              validation_data=val_generator,
                              validation_steps=ceil(n_val_samples/batch_size))

model_name = 'ssd7'
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))
print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()

# TEST
import cv2

test_ouput = pd.read_csv('./data/sample-submission.csv')
for i in range(10000):
  filename = './data/test/'+str(i+1)+'.jpg'
  X = cv2.imread(filename)
  X = np.expand_dims(X, 0)
  y_pred = model.predict(X)
  y_pred_decoded = decode_y2(y_pred, confidence_thresh=0.4, iou_threshold=0.4, top_k='all',
                             input_coords='centroids', normalize_coords=False, img_height=None, img_width=None)
  if len(y_pred_decoded[0]) == 0: label = 'unknown'; print(label);
  else:box = y_pred_decoded[0][0]; label = '{}: {:.2f}'.format(int(box[0]), box[1])
  label = label.split(':')[0]
  print(label)
  test_ouput._set_value(i, 'Number', label)
test_ouput.to_csv("./data/test_output.csv", index=False)