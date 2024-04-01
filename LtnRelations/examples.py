from main import *
import torch
import ltn
from spatial_relations import *
from generate_equally_spaced_bb import generate_eq_spaced_bbs_in_, plot_all_bb_with_names

# Definição das bounding boxes usando o script do Odalísio

bb1 = torch.tensor([0.5, 0.5, 0.2, 0.6])
boxes = generate_eq_spaced_bbs_in_(bb1)
plot_all_bb_with_names(bb1, boxes)

# Verificando o tipo dos dados em boxes e convertendo para tensores se necessário

bb_u = torch.tensor(boxes["U"]) if isinstance(boxes["U"], list) else boxes["U"]
bb_w = torch.tensor(boxes["W"]) if isinstance(boxes["W"], list) else boxes["W"]
bb_x = torch.tensor(boxes["X"]) if isinstance(boxes["X"], list) else boxes["X"]
bb_y = torch.tensor(boxes["Y"]) if isinstance(boxes["Y"], list) else boxes["Y"]
bb_a = torch.tensor(boxes["A"]) if isinstance(boxes["A"], list) else boxes["A"]
bb_z = torch.tensor(boxes["Z"]) if isinstance(boxes["Z"], list) else boxes["Z"]
bb_r = torch.tensor(boxes["R"]) if isinstance(boxes["R"], list) else boxes["R"]

"""
Foram implementados esses axiomas com as letras alteradas para fazer sentido com o data set: 

    outSideRightBelow(W,Z) ← d(Z,X),left(Z,W),above(Z,W) 
    outBelow(X,Z) ← d(Z,X),above(Z,X) 
    outSideLeftBelow(Y,Z) ← d(Z,Y),left(Y,Z),above(Y,Z) 
    outSideLeft(R,Z) ← d(Z,R),left(R,Z) 
    in(Z, A) ← p(A, Z)

    Assume-se que a box menor é a Z e a box que contem todas as outras é a A

"""

# Para outSideRightBelow = True

result = OutSideRightBelow()(bb_w, bb_z)
print("Is W outside right below Z?", result.item())

# Para outSideRightBelow = False

result = OutSideRightBelow()(bb_z, bb_w)
print("Is Z outside right below W?", result.item())

# Para OutBelow = True

result = OutBelow()(bb_x,bb_z)
print("Is X outside below Z?", result.item())

# Para OutBelow = False

result = OutBelow()(bb_z,bb_x)
print("Is Z outside below X?", result.item())

# Para OutSideLeftBelow = True

result = OutSideLeftBelow()(bb_y,bb_z)
print("Is Y outside left below Z?", result.item())

# Para OutSideLeftBelow = False

result = OutSideLeftBelow()(bb_z,bb_y)
print("Is Z outside left below Y?", result.item())

# Para OutSideLeft = True

result = OutSideLeft()(bb_r,bb_z)
print("Is R outside left Z?", result.item())

# Para OutSideLeft = False

result = OutSideLeft()(bb_z,bb_r)
print("Is Z outside left R?", result.item())

# Para In(contains) = True

result = Contains()(bb_z,bb_a)
print("Is Z in A?", result.item())

# Para In(contains) = False

result = Contains()(bb_a,bb_z)
print("Is A in Z?", result.item())
