import torch

def extract_coordinates(bb):
    return bb.unbind(dim=-1)

class LeftPredicate(torch.nn.Module):
    def forward(self, bb1, bb2):
        xmin1, _, xmax1, _ = extract_coordinates(bb1)
        xmin2, _, xmax2, _ = extract_coordinates(bb2)
        return (xmin1-(xmax1/2) + xmax1 < xmin2-(xmax2/2)).float()

class RightPredicate(torch.nn.Module):
    def forward(self, bb1, bb2):
        xmin1, _, xmax1, _ = extract_coordinates(bb1)
        xmin2, _, xmax2, _ = extract_coordinates(bb2)
        return (xmin1-(xmax1/2) > xmin2-(xmax2/2) + xmax2).float()


class AbovePredicate(torch.nn.Module):
    def forward(self, bb1, bb2):
        _, ymin1, _, ymax1 = extract_coordinates(bb1)
        _, ymin2, _, ymax2 = extract_coordinates(bb2)
        return (ymin1-(ymax1/2) + ymax1 < ymin2-(ymax2/2)).float()

class BelowPredicate(torch.nn.Module):
    def forward(self, bb1, bb2):
        _, ymin1, _, ymax1 = extract_coordinates(bb1)
        _, ymin2, _, ymax2 = extract_coordinates(bb2)
        return (ymin1-(ymax1/2) > ymin2-(ymax2/2) + ymax2).float()

# Script da milena convertido para pytorch

def O_pytorch(bb1, bb2):
    bb1_min = bb1[..., :2] - bb1[..., 2:] / 2
    bb1_max = bb1[..., :2] + bb1[..., 2:] / 2
    bb2_min = bb2[..., :2] - bb2[..., 2:] / 2
    bb2_max = bb2[..., :2] + bb2[..., 2:] / 2
    
    overlap_x = torch.min(bb1_max[..., 0], bb2_max[..., 0]) > torch.max(bb1_min[..., 0], bb2_min[..., 0])
    overlap_y = torch.min(bb1_max[..., 1], bb2_max[..., 1]) > torch.max(bb1_min[..., 1], bb2_min[..., 1])
    
    return overlap_x & overlap_y

def PO_pytorch(bb1, bb2):
    bb1_min = bb1[..., :2] - bb1[..., 2:] / 2
    bb1_max = bb1[..., :2] + bb1[..., 2:] / 2
    bb2_min = bb2[..., :2] - bb2[..., 2:] / 2
    bb2_max = bb2[..., :2] + bb2[..., 2:] / 2
    
    overlap_x = torch.min(bb1_max[..., 0], bb2_max[..., 0]) > torch.max(bb1_min[..., 0], bb2_min[..., 0])
    overlap_y = torch.min(bb1_max[..., 1], bb2_max[..., 1]) > torch.max(bb1_min[..., 1], bb2_min[..., 1])
    
    not_contained_x = (bb2_max[..., 0] > bb1_min[..., 0]) & (bb2_min[..., 0] < bb1_max[..., 0])
    not_contained_y = (bb2_max[..., 1] > bb1_min[..., 1]) & (bb2_min[..., 1] < bb1_max[..., 1])
    
    return overlap_x & overlap_y & (not_contained_x | not_contained_y)

def D_pytorch(bb1, bb2):
    bb1_min = bb1[..., :2] - bb1[..., 2:] / 2
    bb1_max = bb1[..., :2] + bb1[..., 2:] / 2
    bb2_min = bb2[..., :2] - bb2[..., 2:] / 2
    bb2_max = bb2[..., :2] + bb2[..., 2:] / 2
    
    disjoint_x = (bb1_max[..., 0] <= bb2_min[..., 0]) | (bb2_max[..., 0] <= bb1_min[..., 0])
    disjoint_y = (bb1_max[..., 1] <= bb2_min[..., 1]) | (bb2_max[..., 1] <= bb1_min[..., 1])
    
    return disjoint_x | disjoint_y
