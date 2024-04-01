import cv2
import numpy as np
import torch
from LtnRelations.spatial_relations import *
from LtnRelations.ltnRelations import *
from detect import detect_objects

def convert_bb_for_ltn(x_min, y_min, x_max, y_max):
    centro_x = (x_min + x_max) / 2.0
    centro_y = (y_min + y_max) / 2.0
    largura = x_max - x_min
    altura = y_max - y_min
    return torch.tensor([centro_x, centro_y, largura, altura])

def describe_relation(bb1, bb2):
    out_side_left_below = OutSideLeftBelow()
    out_below = OutBelow()
    out_side_right_below = OutSideRightBelow()
    contains = Contains()
    out_side_left = OutSideLeft()
    
    if out_side_left_below(bb1, bb2).item():
        return f"Significa um cenário de um passe ou recepção no basquete, ou um cabeceio no futebol."
    elif out_below(bb1, bb2).item():
        return f"Tanto no basquete quando no futebol representam casos bem específicos, como um jogador longe da bola se preparando para bater uma falta ou um penalti."
    elif contains(bb1, bb2).item():
        return f"Mostra o jogador com controle da bola e pode tomar diversas ações, como passe ou chute"
    elif out_side_right_below(bb1, bb2).item():
         return f"Mostra um cenário de um passe ou uma recepção no basquete, ou um cabeceio no futebol."
    elif out_side_left(bb1, bb2).item():
        return f"Mostra um cenário que o jogador possui a posse da bola em uma situação de bola parada, como um tiro de meta ou pode definir uma aproximação de um jogador X no jogador Y que possui a posse da bola. "
    
    return "Relação não especificada"


def process_image(image_path, show=False):
    detections = detect_objects(image_path, show)

    object_labels = ['animal', 'dog', 'cat', 'sports ball']

    person_boxes = []
    object_boxes = []

    for det in detections:
        x_min, y_min, x_max, y_max, label = det
        bb = convert_bb_for_ltn(x_min, y_min, x_max, y_max)

        if label == 'person':
            person_boxes.append((bb, label))
        elif label in object_labels:
            object_boxes.append((bb, label))

    if person_boxes and object_boxes:
        for person_bb, person_label in person_boxes:
            for object_bb, object_label in object_boxes:
                description = describe_relation(object_bb, person_bb)
                print(description)


if __name__ == '__main__':
    process_image('OutBelow.jpg', show=True)

