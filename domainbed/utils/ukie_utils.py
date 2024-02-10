import torch
from torch import nn
import torch.nn.functional as F


loss_mapping = {
    "mse": nn.MSELoss(),
    "ce": nn.CrossEntropyLoss(),
    "kl": nn.KLDivLoss(reduction="batchmean")
}


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val.item()


class UKIELoss(nn.Module):
    def __init__(self,
                 rec_loss: str = None,
                 cls_loss: str = None,
                 dis_loss: str = None):
        super().__init__()

        self.rec_loss = nn.MSELoss() if not rec_loss else loss_mapping[rec_loss]
        self.cls_loss = nn.CrossEntropyLoss() if not cls_loss else loss_mapping[cls_loss]
        self.dis_loss = nn.KLDivLoss() if not dis_loss else loss_mapping[dis_loss]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    def forward(self, args, logits, rec, inv, var, img, label):
        cls_loss = self.cls_loss(logits, label)
        rec_loss = self.rec_loss(rec, img)
        psnr_val = 20 * torch.log10(torch.max(img) / torch.sqrt(rec_loss))

        # init inv_loss // var_loss
        inv_loss = 0
        var_loss = 0
        # check + get label list
        u_label = list(set(label.to(torch.device("cpu")).numpy()))
        # Loop for all labels in label list
        for cls in u_label:
            inv_cls = torch.absolute(inv[label == cls] - inv[label == cls].mean(dim=0))
            var_cls = torch.absolute(var[label == cls] - var[label == cls].mean(dim=0))

            std_x = torch.mean(inv_cls) + 0.0001  # torch.sqrt(inv.var(dim=0) + 0.0001)
            std_y = torch.mean(var_cls) + 0.0001  # torch.sqrt(var.var(dim=0) + 0.0001)

            inv_loss += torch.mean(F.relu(std_x))
            var_loss += torch.mean(F.relu(1 - std_y))

        inv_loss = inv_loss / len(u_label)
        var_loss = var_loss / len(u_label)

        total_loss = args.rec_coeff * rec_loss + \
                     args.inv_coeff * inv_loss + \
                     args.var_coeff * var_loss + \
                     args.cls_coeff * cls_loss

        irep_loss = args.inv_coeff * inv_loss + \
                    args.cls_coeff * cls_loss

        return {
            "cls_loss": cls_loss,
            "rec_loss": rec_loss,
            "psnr_loss": psnr_val,
            "inv_loss": inv_loss,
            "var_loss": var_loss,
            "irep_loss": irep_loss,
            "total_loss": total_loss
        }
