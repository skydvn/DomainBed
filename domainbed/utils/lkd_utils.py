import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.special import softmax


class CReliability:
    def __init__(self, data_loader, num_domain, num_classes, teacher_model):
        self.num_classes = num_classes
        # Data Preprocessing
        self.data_loader = data_loader
        self.num_domain = num_domain
        self.teacher_model = teacher_model
        self.weighting_temperature = 20
        self.teacher_class_score = self.scoringAUC()

    # def scoringAUC(self):
    #     teacher_class_score = []
    #     for i_domain in range(self.num_domain):
    #         predictions = self.teacher_model[i_domain](self.data_x)
    #
    #         rounded_predictions = torch.argmax(predictions, axis=-1)
    #         rounded_predictions_oh = F.one_hot(rounded_predictions, num_classes=self.num_classes).to('cpu').numpy()
    #
    #         teacher_class_score.append(
    #             roc_auc_score(self.data_y, rounded_predictions_oh, average=None, multi_class="ovr"))
    #     return teacher_class_score

    def scoringAUC(self):
        teacher_class_score = []
        for i_domain in range(self.num_domain):
            all_predictions = []
            all_true_labels = []

            # Assuming self.data_x is a DataLoader or similar object that loads data in batches
            for x_batch, y_batch in self.data_loader:
                predictions = self.teacher_model[i_domain](x_batch)
                y_batch = F.one_hot(y_batch, num_classes=self.num_classes).to('cpu').numpy()

                # Convert to probabilities if predictions are logits
                probabilities = F.softmax(predictions, dim=-1)

                all_predictions.append(probabilities.detach().cpu().numpy())
                all_true_labels.append(y_batch)

            # Concatenate all batch results
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_true_labels = np.concatenate(all_true_labels, axis=0)

            # Compute AUC-ROC score for each class and store it
            auc_scores = roc_auc_score(all_true_labels, all_predictions, average=None, multi_class="ovr")
            teacher_class_score.append(auc_scores)
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
    aligned_data_teacher = [torch.tensor([]) for _ in range(num_classes)]
    aligned_label_teacher = [torch.tensor([]) for _ in range(num_classes)]
    for x, y_mu, y_true in pseudo_dataset:
        if aligned_data_teacher[y_mu].nelement() == 0:
            aligned_data_teacher[y_mu] = x.unsqueeze(0)  # Add a new axis to make it a 2D tensor
            aligned_label_teacher[y_mu] = y_true.unsqueeze(0)
        else:
            aligned_data_teacher[y_mu] = torch.cat((aligned_data_teacher[y_mu], x.unsqueeze(0)), dim=0)
            aligned_label_teacher[y_mu] = torch.cat((aligned_label_teacher[y_mu], y_true.unsqueeze(0)), dim=0)

    return aligned_data_teacher, aligned_label_teacher
