from functools import partial
import json
# import logging
import os
import glob
import statistics
import time

from matplotlib import pyplot

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

from fastai.vision.all import load_learner, Path, PILImage, vision_learner, accuracy, show_image, tuplify, detuplify

from model_trainer.dataloader import get_loader
from model_trainer.conf import DATASET_PATH, EVAL_CONFS_JSON, EVAL_PREDICTION_JSON, MODEL_ID



def predict(self, item, rm_type_tfms=None, with_input=False):
    dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
    start_time = time.time()
    inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
    i = getattr(self.dls, 'n_inp', -1)
    inp = (inp,) if i==1 else tuplify(inp)
    dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
    dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
    res = dec_targ,dec_preds[0],preds[0], start_time
    if with_input: res = (dec_inp,) + res
    return res



class ClsEvaluator(object):

    def __init__(self, eval_images_dir, base_model, fine_tune_name):
        self.eval_confs = None
        self.setup_eval_confs()
        self.eval_images_dir = eval_images_dir
        self.images_paths = glob.glob(os.path.join(eval_images_dir, '*.png'))
        self.setup_model(base_model, fine_tune_name)
        self.predicted_values = {}
        self.true_positives_events = []
        self.true_negatives_events = []
        self.false_positives_events = []
        self.false_negatives_events = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1score = 0
        self.debug = False
        self.latencies = []

    def setup_model(self, base_model, fine_tune_name):
        self.device = torch.device('cpu')
        dls = get_loader(DATASET_PATH)
        # import ipdb; ipdb.set_trace()
        self.learn = vision_learner(dls, base_model, metrics=accuracy)
        # , cpu=True)
        self.learn.load(fine_tune_name, device=self.device)
        self.learn.dls.device = torch.device('cpu')
        # self.learn.model.cuda()
        self.model = self.learn.model.to(self.device)
        # self.model = self.learn.model[0].to(self.device)
        self.model.eval()
        self.learn.predict = partial(predict, self.learn)


    def setup_eval_confs(self):
        self.eval_confs = None
        with open(EVAL_CONFS_JSON, 'r') as f:
            self.eval_confs = json.load(f)

    def calculate_accuracy(self):
        true_positives = len(self.true_positives_events)
        true_negatives = len(self.true_negatives_events)
        false_positives = len(self.false_positives_events)
        false_negatives = len(self.false_negatives_events)

        nominator = true_positives + true_negatives
        denominator = true_positives + true_negatives + false_positives + false_negatives
        accuracy = nominator / denominator
        return accuracy

    def calculate_precision(self):
        true_positives = len(self.true_positives_events)
        false_positives = len(self.false_positives_events)

        return true_positives / (true_positives + false_positives)

    def calculate_recall(self):
        true_positives = len(self.true_positives_events)
        false_negatives = len(self.false_negatives_events)
        return true_positives / (true_positives + false_negatives)

    def calculate_f1_score(self, precision, recall):
        nominator = 2 * (precision * recall)
        denominator = precision + recall
        f_1 = nominator / denominator
        return f_1

    def calculate_metrics(self):
        self.accuracy = self.calculate_accuracy()
        self.precision = self.calculate_precision()
        self.recall = self.calculate_recall()
        self.f_score = self.calculate_f1_score(self.precision, self.recall)
        avg_latency = statistics.mean(self.latencies)
        std_latency = statistics.stdev(self.latencies)
        results = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f_score,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'throughput': 1/avg_latency,
        }
        return results

    def run(self, threshold=0.5, recreate_predict=False):
        run_predict = not os.path.exists(EVAL_PREDICTION_JSON) or recreate_predict
        run_predict = True

        if run_predict is False:
            with open(EVAL_PREDICTION_JSON, 'r') as f:
                self.predicted_values = json.load(f)

        total_to_eval = self.eval_confs['Total_Non_Ignored_Frames']
        total_images = len(self.images_paths)
        total_eval = 0
        for proc_index, image_path in enumerate(self.images_paths[:1000]):
        # for proc_index, image_path in enumerate(self.images_paths):
            image_id = os.path.basename(image_path).split('.')[0]
            # if image_id not in self.predicted_values:
            #     continue

            # if run_predict is False:
            if run_predict is True:
                class_probs = self.model_predict(image_path)
                # try:
                #     np.testing.assert_almost_equal(
                #         self.predicted_values[image_id], class_probs,
                #         decimal=4
                #     )
                # except AssertionError as e:
                #     print(e)
                #     print(f'previous: {self.predicted_values[image_id]}, new : {class_probs}')
                if recreate_predict is True:
                    self.predicted_values[image_id] = class_probs
            else:
                class_probs = self.predicted_values[image_id]

            non_oi_prob, has_oi_prob = class_probs

            positive_pred = has_oi_prob > threshold

            processed = 1
            if image_id in self.eval_confs['TP']:
                if positive_pred:
                    self.true_positives_events.append(image_id)
                else:
                    self.false_positives_events.append(image_id)
            elif image_id in self.eval_confs['TN']:
                if not positive_pred:
                    self.true_negatives_events.append(image_id)
                else:
                    self.false_negatives_events.append(image_id)
            else:
                processed = 0

            total_eval += processed
            if self.debug and proc_index % 500 == 0:
                print(f'i: {proc_index + 1}/{total_images}. total_eval: {total_eval/total_to_eval * 100}%')

        if recreate_predict is True:
            with open(EVAL_PREDICTION_JSON, 'w') as f:
                json.dump(self.predicted_values, f)
        if self.debug:
            print(f'total_eval: {total_eval}. total_to_eval {total_to_eval}. total_images: {total_images}.')
        results = {'threshold': threshold}
        results.update(self.calculate_metrics())
        return results

    def model_predict(self, image_path):
        img = PILImage.create(image_path)

        with self.learn.no_bar(), self.learn.no_mbar(), self.learn.no_logging():
            start_time = time.time()
            # Make predictions on the image
            predicted_class, _, class_probs, start_time = self.learn.predict(img)
            class_probs = class_probs.tolist()
            end_time = time.time()
            predict_time = (end_time - start_time)
            self.latencies.append(predict_time)

        return class_probs

    # def model_predict(self, image_path):
    #     img = PILImage.create(image_path)
    #     dl = self.learn.dls.test_dl([img], rm_type_tfms=None, num_workers=0)
    #     import ipdb; ipdb.set_trace()
    #     image = dl.one_batch()[0]
    #     with torch.no_grad():
    #         start_time = time.time()
    #         prediciton = self.model(image).squeeze(0).softmax(0)
    #         class_probs = prediciton.tolist()
    #         end_time = time.time()
    #         predict_time = (end_time - start_time)
    #         self.latencies.append(predict_time)

    #     return class_probs

    # def model_predict(self, image_path):
    #     """trying equivalent transf"""
    #     img = PILImage.create(image_path)
    #     # dl = self.learn.dls.test_dl([img], rm_type_tfms=None, num_workers=0)
    #     # preprocess = transforms.Compose([
    #     #     transforms.Resize(256),
    #     #     transforms.CenterCrop(224),
    #     #     transforms.ToTensor(),
    #     #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     # ])


    #     # try this with old model.
    #     preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])

    #     start_time = time.time()
    #     input_tensor = preprocess(img)
    #     input_batch = input_tensor.unsqueeze(0)
    #     with torch.no_grad():
    #         prediciton = self.model(input_batch).squeeze(0).softmax(0)
    #         class_probs = prediciton.tolist()
    #         end_time = time.time()
    #         predict_time = (end_time - start_time)
    #         self.latencies.append(predict_time)

    #     return class_probs



    # def model_predict(self, image_path):
    #     img = PILImage.create(image_path)
    #     # preprocess = models.MobileNet_V3_Large_Weights.DEFAULT.transforms().to(self.device)

    #     with torch.no_grad():
    #         dl = self.learn.dls.test_dl([img], rm_type_tfms=None, num_workers=0)
    #         start_time = time.time()
    #         image = dl.one_batch()[0]  # Move the image to CPU
    #         # image = preprocess(img).unsqueeze(0)
    #         prediciton = self.model(image).squeeze(0).softmax(0)
    #         class_probs = prediciton.tolist()
    #         end_time = time.time()
    #         predict_time = (end_time - start_time)
    #         self.latencies.append(predict_time)

    #     return class_probs


    # def model_predict(self, image_path):
    #     img = PILImage.create(image_path)
    #     import ipdb; ipdb.set_trace()
    #     dl = self.learn.dls.test_dl([img], rm_type_tfms=None, num_workers=0)
    #     image = dl.one_batch()[0]
    #     with torch.no_grad():
    #         start_time = time.time()
    #         prediciton = self.model(image).squeeze(0).softmax(0)
    #         class_probs = prediciton.tolist()
    #         end_time = time.time()
    #         predict_time = (end_time - start_time)
    #         self.latencies.append(predict_time)

    #     return class_probs


if __name__ == '__main__':
    # threshold = float(sys.argv[1])

    EVAL_PREDICTION_JSON = EVAL_PREDICTION_JSON.replace('.json', '-fastai.json')
    eval_images_dir = '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-Q-1'

    # # Disable logging for fastai and its dependencies
    # logging.getLogger('fastai').setLevel(logging.CRITICAL)
    # logging.getLogger('torch').setLevel(logging.CRITICAL)
    # logging.getLogger('fastai').setLevel(logging.CRITICAL)
    # logging.getLogger('torchvision').setLevel(logging.CRITICAL)

    evaluator = ClsEvaluator(eval_images_dir, models.mobilenet_v3_large, 'cls_model_first')
    evaluator.debug = True
    threshold = 0.5
    for threshold in [0.5, 0.6, 0.65, 0.75, 0.8, 0.95]:
        res = evaluator.run(threshold=threshold, recreate_predict=False)
        print(json.dumps(res, indent=4))
        break
    # evaluator.save()
