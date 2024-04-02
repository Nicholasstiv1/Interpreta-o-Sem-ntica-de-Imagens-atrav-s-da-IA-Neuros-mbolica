import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
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
    out_side_right_below = OutSideRightBelow()
    out_below = OutBelow()
    contains = Contains()
    out_side_left = OutSideLeft()
    
    if out_side_left_below(bb1, bb2).item():
        return "Significa um cenário de um passe ou recepção no basquete, ou um cabeceio no futebol.", "OutSideLeftBelow"
    elif out_side_right_below(bb1, bb2).item():
        return "Mostra um cenário de um passe ou uma recepção no basquete, ou um cabeceio no futebol.", "OutSideRightBelow"
    elif out_below(bb1, bb2).item():
        return "Tanto no basquete quando no futebol representam casos bem específicos, como um jogador longe da bola se preparando para bater uma falta ou um penalti.", "OutBelow"
    elif contains(bb1, bb2).item():
        return "Mostra o jogador com controle da bola e pode tomar diversas ações, como passe ou chute", "Contains"
    elif out_side_left(bb1, bb2).item():
        return "Mostra um cenário que o jogador possui a posse da bola em uma situação de bola parada, como um tiro de meta ou pode definir uma aproximação de um jogador X no jogador Y que possui a posse da bola.", "OutSideLeft"
    
    return "Relação não especificada", "Unknown"


def process_image(image_path, show=False):
    detections = detect_objects(image_path, show)
    object_labels = ['animal', 'dog', 'cat', 'sports ball']

    person_boxes = []
    object_boxes = []
    relations_data = []
    relations_seen = set()

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
                person_bb_tuple = tuple(person_bb.tolist())
                object_bb_tuple = tuple(object_bb.tolist())

                relation_id_person_object = (person_bb_tuple, object_bb_tuple)
                relation_id_object_person = (object_bb_tuple, person_bb_tuple)

                desc_person_object, type_person_object = describe_relation(person_bb, object_bb)
                desc_object_person, type_object_person = describe_relation(object_bb, person_bb)

                if type_person_object != "Unknown" and relation_id_person_object not in relations_seen:
                    relations_data.append({
                        'person_bb': person_bb,
                        'object_bb': object_bb,
                        'relation_type': type_person_object,
                        'description': desc_person_object
                    })
                    relations_seen.add(relation_id_person_object)

                if type_object_person != "Unknown" and relation_id_object_person not in relations_seen:
                    relations_data.append({
                        'person_bb': person_bb,
                        'object_bb': object_bb,
                        'relation_type': type_object_person,
                        'description': desc_object_person
                    })
                    relations_seen.add(relation_id_object_person)

    return relations_data

def validate_model(image_paths, annotations):
    predictions = []
    ground_truths = []

    for image_path, annotation in zip(image_paths, annotations):
        relations_data = process_image(image_path)
        for data in relations_data:
            predictions.append(data['relation_type'])
            ground_truths.append(annotation)

    labels = sorted(set(predictions + ground_truths))

    conf_matrix = confusion_matrix(ground_truths, predictions, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predição')
    plt.savefig('matriz_confusao.png')
    plt.close()

    report = classification_report(ground_truths, predictions, labels=labels, output_dict=True)
    with open('relatorio_classificacao.txt', 'w') as f:
        f.write(classification_report(ground_truths, predictions, labels=labels))


if __name__ == '__main__':
    image_paths = ['images/OutBelow.jpg', 'images/OutLeft.jpg','images/In.jpg','images/Sides.jpg','images/SidesReverse.jpg']  
    annotations = ['OutBelow', 'OutSideLeft','Contains','OutSideRightBelow','OutSideLeftBelow']  

    validate_model(image_paths, annotations)
