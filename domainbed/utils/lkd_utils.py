import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.special import softmax


class CReliability:
    def __init__(self, data_x, data_y, num_domain, num_classes, teacher_model):
        self.num_classes = num_classes
        # Data Preprocessing
        self.data_x = data_x
        self.data_y = F.one_hot(data_y, num_classes=self.num_classes).to('cpu').numpy()
        self.num_domain = num_domain
        self.teacher_model = teacher_model
        self.weighting_temperature = 20
        self.teacher_class_score = self.scoringAUC()

    def scoringAUC(self):
        teacher_class_score = []
        for i_domain in range(self.num_domain):
            predictions = self.teacher_model[i_domain](self.data_x)

            rounded_predictions = torch.argmax(predictions, axis=-1)
            rounded_predictions_oh = F.one_hot(rounded_predictions, num_classes=self.num_classes).to('cpu').numpy()

            teacher_class_score.append(
                roc_auc_score(self.data_y, rounded_predictions_oh, average=None, multi_class="ovr"))
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


def dataAlignment(pseudo_dataset, num_classes):
    # Init aligned_data_teacher
    init_arr = np.array([])
    aligned_data_teacher = {}
    aligned_label_teacher = {}

    for i in range(0, num_classes):
        aligned_data_teacher[i] = init_arr

    for x, y_mu, y_true in pseudo_dataset:
        for i in range(0, num_classes):
            if i == y_mu:
                if aligned_data_teacher[i].size == 0:
                    aligned_data_teacher[i] = x
                    aligned_label_teacher[i] = y_true
                else:
                    aligned_data_teacher[i] = np.vstack((aligned_data_teacher[i], x))
                    aligned_label_teacher[i] = np.vstack((aligned_label_teacher[i], y_true))
    return aligned_data_teacher, aligned_label_teacher
