import torch
import ltn
from .spatial_relations import *
from .generate_equally_spaced_bb import generate_eq_spaced_bbs_in_, plot_all_bb_with_names

class OutSideRightBelow(torch.nn.Module):
    def __init__(self):
        super(OutSideRightBelow, self).__init__()
        self.right_pred = RightPredicate()
        self.below_pred = BelowPredicate()

    def forward(self, bb1, bb2):
        not_contained = D_pytorch(bb1, bb2).bool()
        is_right = self.right_pred(bb1, bb2).bool()
        is_below = self.below_pred(bb1, bb2).bool()

        return not_contained & is_right & is_below

class OutBelow(torch.nn.Module):
    def __init__(self):
        super(OutBelow, self).__init__()
        self.below_pred = BelowPredicate()

    def forward(self, bb1, bb2):
        is_below = self.below_pred(bb1, bb2).bool()
        not_contained = D_pytorch(bb1, bb2).bool()

        return is_below & not_contained

class OutSideLeftBelow(torch.nn.Module):
    def __init__(self):
        super(OutSideLeftBelow, self).__init__()
        self.left_pred = LeftPredicate()
        self.below_pred = BelowPredicate() 

    def forward(self, bb1, bb2):
        not_contained = D_pytorch(bb1, bb2).bool()
        is_left = self.left_pred(bb1, bb2).bool()
        is_below = self.below_pred(bb1, bb2).bool()

        return is_left & is_below & not_contained

class OutSideLeft(torch.nn.Module):
    def __init__(self):
        super(OutSideLeft, self).__init__()
        self.left_pred = LeftPredicate()

    def forward(self, bb1, bb2):
        not_contained = D_pytorch(bb1, bb2).bool()
        is_left = self.left_pred(bb1, bb2).bool()

        return is_left & not_contained

class Contains(torch.nn.Module):
    def __init__(self):
        super(Contains, self).__init__()

    def forward(self, bb1, bb2):
        bb1_min = bb1[..., :2] - bb1[..., 2:] / 2
        bb1_max = bb1[..., :2] + bb1[..., 2:] / 2
        bb2_min = bb2[..., :2] - bb2[..., 2:] / 2
        bb2_max = bb2[..., :2] + bb2[..., 2:] / 2
        
        contained = torch.all(bb2_min >= bb1_min) and torch.all(bb2_max <= bb1_max)

        return contained
