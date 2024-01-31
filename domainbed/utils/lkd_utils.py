import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.special import softmax


class CReliability:
    def __init__(self, base_samples_region_train, num_domain, num_classes, teacher_model):
        self.base_samples_region_train = base_samples_region_train
        self.num_classes = num_classes
        # Data Preprocessing
        self.processed_data_x = self.base_samples_region_train[0]
        self.processed_data_y = self.base_samples_region_train[1]

        self.teacher_class_score = self.scoringAUC()
        self.num_domain = num_domain
        self.teacher_model = teacher_model
        self.weighting_temperature = 20

    def scoringAUC(self):
        teacher_class_score = []
        for i_domain in range(self.num_domain):
            predictions = self.teacher_model[i_domain].predict(x=self.processed_data_x, batch_size=32, verbose=False)
            rounded_predictions = np.argmax(predictions, axis=-1)

            rounded_predictions = rounded_predictions.long()
            rounded_predictions_oh = F.one_hot(rounded_predictions, num_classes=self.num_classes)
            rounded_predictions_oh = rounded_predictions_oh.transpose(0, 1)
            # print(self.processed_data_y.shape)
            # print(rounded_predictions_oh.shape)

            teacher_class_score.append(
                roc_auc_score(self.processed_data_y, rounded_predictions_oh, average=None, multi_class="ovr"))
        return teacher_class_score

    def weightedClass(self):
        teacher_class_score_softmax = []
        teacher_class_score = []
        for i in range(0, len(self.teacher_class_score)):
            teacher_class_score.append(self.teacher_class_score[i])
        teacher_class_score_transpose = np.transpose(teacher_class_score)
        for class_index in range(self.num_classes):
            teacher_class_score_softmax.append(
                softmax(teacher_class_score_transpose[class_index] * self.weighting_temperature))
        return np.transpose(teacher_class_score_softmax)
