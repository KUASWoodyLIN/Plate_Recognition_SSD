import cv2
import numpy as np
import pandas as pd
from Plate_Recognition_SSD.ssd_box_encode_decode_utils import decode_y2
from keras.models import load_model
ids=['background', 'AGV7939', 'X59329', 'ADB2531', '8586VN', '5786YZ', '9256A9', '8673K9', 'AFG1929',
     '7050EB', '5199T2', '2E1723', '8381YJ', '6250M3', '1553ZS', '0592VE', '3827J8', 'RAJ8668',
     '0198YZ', '0871RG', '6237EJ', '5K5155', '0719DE', 'APM9171', 'AHF2353', '8128G5', '8672K9',
     'RAD8819', '3863J7', '0663UX', 'V63698', '6D9569', 'AAQ6006', '1702YC', '8C3676', '6791H8',
     'AFG8307', '8210M5', '333633', '0697QN', '1550ZS', '0573VN', '1863J6', '6790ZW', 'AGP2206',
     '5823DY', 'AGV8338', '5902VF', '6A8338', '1850YN', 'ALB3772', 'AGT7210', '8008DX', '2975A5',
     'RBD8610', '2E0325', 'AAP2926', '5119J3', '4571H6', '7285UT', 'AAP2927', '7679J2', 'AGE5239',
     'V22999', '8E8338', '6150J5', 'AFJ1165', 'N88450', '5255J6', 'AFG0386', '5005EU', '8979YD',
     'AAL8668', 'RBB8058', 'AHB5826', 'AFJ8512', '7821VH', '5592MP', '7267YA', '6613J7', '9590VE',
     '7U9123', '8671K9', '988833', '8195VB', '9929UX', 'AFJ1121', '829W6', 'ABY2895', 'AGS0100',
     'AGC6259', '6522QA', 'AKP1637', '7D8953', '5150D2', 'ZM5831', 'AGT7230', '3719YJ', 'AGL1853',
     '8663J8', 'AGF9119', '9233L7', '3838EU', '9023L7', '8208S3', '1508L8', '863J6', '0209J3',
     '2712A3', 'AFF6790', '479922', 'RAN9181', 'AKZ1266', '3861J2', '3165J3', 'AFG7232', 'AAM7252',
     'AFK3965', 'DK6266', '5335QT', '3R3337', '8516DG', '6171KL', '9101ZR', 'APM5527', 'APK5605',
     '6336VS', 'RAA7500', 'AFJ8612', '4537EH', '9171QM', 'RAF2380', '9936YC', '3993YG', '2782J9',
     '9168T2', '6908YQ', 'ALG8668', '2820J2', '3079DH', '3812QW', 'ATD1351', '793ZP', 'AJN1621',
     '2689B3', '0685YK', '0631T3', '3G5889', 'AKN9936', '6502KU', 'RAJ6828', '0712QZ', '3A3268',
     'RAQ7557', 'APD1317', '5692S2', 'ABB3877', '9805DX', '8E3310', 'AGL5853', 'AAB0251', 'AKY0053',
     'APC5906', '5C7267', 'AKG3503', '5690EG', '7376EJ', '6153J5', '7770XX', 'AAL7555', 'ANV6770',
     'AKU7838', 'RAB7266', 'AGS9170', '0713QZ', '1663ZT', 'AAD5113', 'APF0059', '5195B7', '8533TU',
     '7868QD', '2786L3', '2393A9', '1113P2', '3N8761', '7873RH', '3163J3', '0303L3', '6173EP',
     'AKW5712', '1289A3', 'AGT7221', 'AGV7935', '2990R8', 'IW4266', 'AQA3337', '6639H8', '8520N6',
     'APC6071', '2790J9', '787CZ', 'AFK3952', '0803QB', 'AGV7929', '7177ET', '3T4466', 'AAY7711',
     '5383YB', 'ADD1227', '3A2795', '2K7429', 'ANV6753', 'AGF9299', '8672VN', 'APC5305', '9119G3',
     'ALH5710', 'ABB9051', '5119J5', '5330YP', 'AKL1185', 'ABD6580', 'T21288', 'AGS9180', '6683YK',
     'ABB9061', 'AMZ3823', 'ACC1111', '6795RH', 'AKN8705', 'AAR8955', 'AGV793', '2783J9', '0756SD',
     'AKK2970', 'RAD9119', 'ABB9182', 'DK8088', '2828LM', '719M2', '8668WW', 'AKT2275', '9889B8',
     '6336A6', 'ANZ7938', 'AGR2100', 'AAR8563', 'ABB9183', '5565MQ', 'ADB2551', 'AFF9159', 'AKP0999',
     '8672KV', 'AGJ1565', 'AGF2753', 'AKW8930', '789A6', 'ABB9175', '0707C', '2K2115', 'AKU5781',
     'AFK3953', '6855ZA']


model = load_model('ssd7_weights_epoch-09_loss-4.2779.h5')
test_ouput = pd.read_csv('./data/sample-submission.csv')
for i in range(10000):
  filename = './data/test/'+str(i+1)+'.jpg'
  X = cv2.imread(filename)
  X = np.expand_dims(X, 0)
  y_pred = model.predict(X)
  y_pred_decoded = decode_y2(y_pred, confidence_thresh=0.4, iou_threshold=0.4, top_k='all',
                             input_coords='centroids', normalize_coords=False, img_height=None, img_width=None)
  if len(y_pred_decoded[0]) == 0: label = 'unknown'; print(label);
  else:box = y_pred_decoded[0][0]; label = '{}: {:.2f}'.format(ids[int(box[0])], box[1])
  label = label.split(':')[0]
  print(label)
  test_ouput._set_value(i, 'Number', label)
test_ouput.to_csv("./data/test_output.csv", index=False)