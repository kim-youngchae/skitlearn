import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import noisereduce as nr

def extract_features(file_name):
    try:
        # raw 파일을 읽어서 numpy array로 변환
        with open(file_name, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.int16)
        sample_rate = 16000  # Assuming a sample rate of 16000 Hz

        # 소음 감소 적용
        reduced_noise_data = nr.reduce_noise(y=raw_data, sr=sample_rate, prop_decrease=0.6)

        mfcc_features = mfcc(reduced_noise_data, samplerate=sample_rate, numcep=13)
        mfcc_mean = np.mean(mfcc_features, axis=0)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None
    return mfcc_mean

def preprocess_data(input_dir):
    features = []
    labels = []
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.raw'):
                file_path = os.path.join(root, file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    file_paths.append(file_path)
                    if "male" in file.lower():
                        labels.append(1)  # male
                    else:
                        labels.append(0)  # female
    return np.array(features), np.array(labels), file_paths

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# 학습 데이터 전처리
train_features, train_labels, _ = preprocess_data('train')
if len(train_features) == 0:
    raise ValueError("No training data found. Please check the 'train' directory and ensure it contains '.raw' files.")
save_to_csv(train_features, 'train_features.csv')
save_to_csv(train_labels, 'train_labels.csv')

# 테스트 데이터 전처리
test_features, test_labels, test_files = preprocess_data('test')
if len(test_features) == 0:
    raise ValueError("No test data found. Please check the 'test' directory and ensure it contains '.raw' files.")
save_to_csv(test_features, 'test_features.csv')
save_to_csv(test_labels, 'test_labels.csv')

print(f"전처리된 학습 데이터를 'train_features.csv' 파일로 저장했습니다.")
print(f"전처리된 테스트 데이터를 'test_features.csv' 파일로 저장했습니다.")
print(f"전처리된 학습 라벨을 'train_labels.csv' 파일로 저장했습니다.")
print(f"전처리된 테스트 라벨을 'test_labels.csv' 파일로 저장했습니다.")

# 학습 및 검증 데이터를 로드
train_df = pd.read_csv('train_features.csv')
train_labels_df = pd.read_csv('train_labels.csv')
test_df = pd.read_csv('test_features.csv')
test_labels_df = pd.read_csv('test_labels.csv')

# 데이터 크기 출력
print(f"Number of training samples: {len(train_df)}")
print(f"Number of test samples: {len(test_df)}")

# 학습 데이터를 numpy 배열로 변환
X_train_full = train_df.values
y_train_full = train_labels_df.values.ravel()

if len(X_train_full) == 0:
    raise ValueError("The training set is empty. Please check your data preprocessing steps.")

# 학습 데이터를 학습 세트와 검증 세트로 분할
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Gaussian Mixture Model 학습
gmm = GaussianMixture(n_components=2, random_state=42)  # n_components는 실제 데이터에 맞게 조정 필요
gmm.fit(X_train)

# 학습 데이터에 대한 예측
y_train_pred = gmm.predict(X_train)
y_val_pred = gmm.predict(X_val)

# 정확도 계산
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# 테스트 데이터에 대한 예측
X_test = test_df.values
y_test = test_labels_df.values.ravel()
y_test_pred = gmm.predict(X_test)

# 테스트 결과를 CSV 파일로 저장
test_result_df = pd.DataFrame({
    'File_Path': test_files,
    'Actual_Label': y_test,
    'Predicted_Label': y_test_pred
})
test_result_df.to_csv('test_results.csv', index=False)

print("Test results have been saved to 'test_results.csv'.")
