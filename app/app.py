#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from datetime import datetime
import os
import string
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from config import config
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

device = "cpu"

SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

app = Flask(__name__, static_url_path="")

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@app.route('/')
def index():
    model, converter, length_for_pred, text_for_pred, opt = loader()
    start_time = time.time()

    AlignCollate_demo = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=opt['PAD'])
    demo_data = RawDataset(root=opt['image_folder'], opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=int(opt['workers']),
        collate_fn=AlignCollate_demo, pin_memory=True)

    get_data = time.time() - start_time

    # predict
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # 最大長予測用
            # torch.cuda.synchronize(device)
            if 'CTC' in opt['Prediction']:
                preds = model(image, text_for_pred)#.log_softmax(2)
                preds = preds.log_softmax(2)
                # 最大確率を選択し、インデックスを文字にデコードします
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # 最大確率を選択し、インデックスを文字にデコードします
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            print('-' * 80)
            print('image_path\tpredicted_labels')
            print('-' * 80)
            for img_name, pred in zip(image_path_list, preds_str):
                if 'Attn' in opt['Prediction']:
                    pred = pred[:pred.find('[s]')]  # 文の終わりトークン（[s]）の後の剪定

                print(f'{img_name}\t{pred}')

        forward_time = time.time() - start_time
        only_infer_time = forward_time - get_data

        print('*' * 80)
        print('get_dta_time:{:.5f}[sec]'.format(get_data))
        print('only_infer_time:{:.5f}[sec]'.format(only_infer_time))
        print('total_time:{:.5f}[sec]'.format(forward_time))
        print('*' * 80)

        img_name = [i[9:] for i in image_path_list]
        items = {}
        for path, pred in zip(img_name, preds_str):
            items[path] = pred


    return render_template('index.html', images=items)

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)


# 参考: https://qiita.com/yuuuu3/items/6e4206fdc8c83747544b
@app.route('/upload', methods=['POST'])
def upload():
    if request.files['image']:
        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        # 保存
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
        save_path = os.path.join(SAVE_DIR, dt_now + ".png")
        cv2.imwrite(save_path, img)


    return redirect('/')


def loader():
    opt = config()
    """ model configuration """
    if 'CTC' in opt['Prediction']:
        converter = CTCLabelConverter(opt['character'])
    else:
        converter = AttnLabelConverter(opt['character'])
    opt['num_class'] = len(converter.character)

    if opt['rgb'] == 3:
         opt['input_channel'] = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt['saved_model'], map_location=device))
    length_for_pred = torch.IntTensor([opt['batch_max_length']] * opt['batch_size']).to(device)
    text_for_pred = torch.LongTensor(opt['batch_size'], opt['batch_max_length'] + 1).fill_(0).to(device)
    model.eval()

    return model, converter, length_for_pred, text_for_pred, opt



if __name__ == '__main__':

    app.debug = True
    app.run(host='0.0.0.0', port=8888)
