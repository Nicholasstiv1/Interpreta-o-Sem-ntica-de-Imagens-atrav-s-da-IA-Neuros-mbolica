import cv2
import numpy as np

def detect_objects(image_path, show=False, new_width=800):
    config_path = 'utils/yolov3.cfg'
    weights_path = 'utils/yolov3.weights'
    names_path = 'utils/coco.names'

    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Não foi possível carregar a imagem de: {image_path}")
        return []

    original_height, original_width = image.shape[:2]

    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    blob = cv2.dnn.blobFromImage(resized_image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers_indices = output_layers_indices.flatten() if output_layers_indices.ndim > 1 else output_layers_indices
    output_layers = [layer_names[i - 1] for i in output_layers_indices]

    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x, center_y, w, h = (detection[0:4] * np.array([new_width, new_height, new_width, new_height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes.flatten():
        (x, y, w, h) = boxes[i]
        label = classes[class_ids[i]]
        color = [int(c) for c in np.random.randint(0, 255, size=3)]
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(resized_image, f"{label}: {confidences[i]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if show:
        cv2.imshow("Image with Bounding Boxes", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    final_detections = []
    for i in indexes.flatten():
        (x, y, w, h) = boxes[i]
        label = classes[class_ids[i]]
        final_detections.append((x, y, x+w, y+h, label))

    return final_detections

