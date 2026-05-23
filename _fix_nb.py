import json

path = r'c:\Users\KuMa989\Desktop\PINN_Katyayani\pinn_problem_setup.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_src = []
        for line in cell['source']:
            if '7 modes, 28 PDEs, 6 BCs' in line:
                line = line.replace('7 modes, 28 PDEs, 6 BCs', '5 PDEs, 22 BCs')
                changed += 1
            new_src.append(line)
        cell['source'] = new_src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Done. Changed {changed} line(s).')
