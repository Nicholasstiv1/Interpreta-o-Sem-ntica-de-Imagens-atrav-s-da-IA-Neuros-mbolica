# Interpretação Semântica de Imagens através da IA Neurosimbólica

## Descrição

Este projeto foca na interpretação semântica de imagens que envolvem ações dinâmicas, com um interesse particular em aplicações esportivas. Utilizando o framework de Logical Tensor Networks (LTNtorch) e os predicados espaciais fornecidos pela biblioteca Spatial-Relations, desenvolvemos um sistema que combina técnicas de aprendizado profundo com lógica simbólica para interpretar bounding boxes em contextos de ação.

O objetivo é ir além da simples detecção de objetos, alcançando um entendimento mais profundo das relações e interações presentes nas cenas. A implementação de regras lógicas em LTN permite o raciocínio sobre as bounding boxes e a inferência de ações possíveis, como pode ser observado em cenários esportivos onde a posição e o movimento dos jogadores em relação à bola são essenciais.

Nesse projeto foram implementados os seguintes axiomas:

* outSideRightBelow(V,A) ← d(A,W),left(A,V),above(A,V) 
* outBelow(W,A) ← d(A,W),above(A,W) 
* outSideLeftBelow(X,A) ← d(A,X),left(X,A),above(A,X) 
* outSideLeft(Y,A) ← d(A,Y),left(Y,A) 
* in(A, Z) ← p(Z, A)

## Começando

O projeto está dividido em duas partes principais, a pasta LtnRelations, que contém a implementação dos axiomas citados definidos anteriormente, e o arquivo main.py, que usa essa implementação para descrever e definir as relações entre bounding boxes geradas numa imagem com o yolo.

### Pré-requisitos

* ltn
* torch
* numpy
* matplotlib
* cv2
* seaborn
* sklearn

Também é necessário o uso do YoloV3. O .cfg e o .names já estão disponíveis no projeto, sendo necessário apenas o .weights, que pode ser adquirido direto do site oficial ou com o download direto:

`wget https://pjreddie.com/media/files/yolov3.weights`

