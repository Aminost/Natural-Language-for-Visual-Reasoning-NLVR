import json

data = []

with open("preprocessed-dataset/preprocessed_train_v3.json") as f:
    lines = json.loads(f.read())

for element in lines:  # 1 line = 1 image
    structured_rep = element["structured_rep"]  # 1 image = 1 structured representation
    for frame in structured_rep:  # 1 structured representation = 3 frames
        shapes = []
        for shape in frame:  # 1 frame = up to 8 shapes
            shapes.append({"type":shape["type"], "color":shape["color"], "size":shape["size"], "x_loc":shape["x_loc"], "y_loc":shape["y_loc"]})
        data.append(shapes)

with open("structured_representation_dev.json", 'w') as f:
    json.dump(data, f)
print("done")
