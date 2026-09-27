import glob
import re

for file in glob.glob('**/*.ipynb', recursive=True):
    with open(file, 'r') as f:
        content = f.read()
    
    paths = re.findall(r'/[a-zA-Z0-9_/-]+\.pt', content)
    if paths:
        print(f"Found in {file}: {paths}")
    
    # also check for /content/drive or /kaggle
    paths2 = re.findall(r'(/content/drive[a-zA-Z0-9_/ -]*|/kaggle[a-zA-Z0-9_/ -]*)', content)
    if paths2:
        print(f"Found in {file}: {paths2}")
