import json

with open('model_evaluation.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        for i, line in enumerate(cell['source']):
            if 'classes = test_dataset.classes' in line:
                cell['source'][i] = 'classes = [class_map["idx_to_class"][str(i)] for i in range(19)]\n'
                print('Updated')

with open('model_evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)