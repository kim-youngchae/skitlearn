import sys
import pandas as pd
import os

def calculate_accuracy(test_file, true_file):
    try:
        # 테스트 파일을 DataFrame으로 읽기
        test_df = pd.read_csv(test_file)
        
        # 정답 라벨 파일 읽기
        if not os.path.exists(true_file):
            print(f"Error: 정답 라벨 파일이 존재하지 않습니다: {true_file}")
            return

        with open(true_file, 'r') as f:
            true_lines = f.readlines()

        # 디버깅: 정답 라벨 파일 내용 출력
        print("정답 라벨 파일 내용:")
        for line in true_lines[:5]:  # 처음 5줄만 출력
            print(line.strip())

        # 테스트 파일과 정답 라벨 파일이 비어 있지 않은지 확인
        if test_df.empty:
            print("Error: 테스트 파일이 비어 있습니다.")
            return
        if not true_lines:
            print("Error: 정답 라벨 파일이 비어 있습니다.")
            return

        # 디버깅: 두 리스트의 길이 출력
        print(f"예측 값의 수: {len(test_df)}")
        print(f"정답 라벨의 수: {len(true_lines)}")
        
        # 예측 값의 수와 정답 라벨의 수가 일치하는지 확인
        if len(test_df) != len(true_lines):
            print("Error: 예측 값의 수가 정답 라벨의 수와 일치하지 않습니다.")
            return

        n = 0
        hit = 0

        for index, row in test_df.iterrows():
            predicted_label = row['Predicted_Label']
            true_label_line = true_lines[n].strip()
            true_label = true_label_line.split()[1]

            if str(predicted_label) == true_label:
                hit += 1

            n += 1

        total = n
        acc = (hit / total) * 100
        print("============ 결과 분석 ===========")
        print(f"테스트 파일: {test_file}")
        print(f"정답 파일: {true_file}")
        print(f"정확도: {acc:.2f}%")
        print(f"정답 개수: {hit}, 총 개수: {total}")
        print("=================================")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <test_file> <true_file>")
    else:
        test_file = sys.argv[1]
        true_file = sys.argv[2]
        calculate_accuracy(test_file, true_file)
