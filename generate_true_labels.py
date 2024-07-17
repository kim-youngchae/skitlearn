import pandas as pd

# true_labels.txt 파일에서 데이터를 읽어옵니다.
with open('true_labels.txt', 'r') as file:
    lines = file.readlines()

# 파일 경로와 라벨을 분리하고, 라벨을 0과 1로 변환합니다.
labels = []
for line in lines:
    parts = line.strip().rsplit(' ', 1)
    if len(parts) == 2:
        _, label = parts
        if label == 'feml':
            labels.append(0)
        elif label == 'male':
            labels.append(1)
        else:
            print(f"Unknown label '{label}' in line: {line}")
    else:
        print(f"Skipping malformed line: {line}")

# 변환된 라벨을 CSV 파일로 출력합니다.
with open('true_labels.csv', 'w') as file:
    for label in labels:
        file.write(f"{label}\n")

print("true_labels.csv 파일이 성공적으로 생성되었습니다.")
