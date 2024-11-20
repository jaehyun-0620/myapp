# import os  # 파일 및 디렉토리 관리를 위한 라이브러리
# import shutil  # 파일 이동 및 복사를 위한 라이브러리
# import tensorflow as tf  # 딥러닝 모델링을 위한 TensorFlow
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.utils import image_dataset_from_directory
# import matplotlib.pyplot as plt


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# # 데이터 경로 설정
# base_dir = 'data/RPS Data/RPS Data'
# train_dir = os.path.join(base_dir, 'train')  # 학습 데이터(train) 폴더 경로
# test_dir = os.path.join(base_dir, 'test')  # 테스트 데이터(test) 폴더 경로
#
# #모델 수정 값
# batch = 32          # 배치
# epoch = 5            # 에폭
# split = 0.20         # 검증 데이터로 사용할 train 데이터 비율
#
# #모델 만들기
# inputs = keras.Input(shape = (224, 224, 3))
# x = layers.Rescaling(1./255)(inputs)
# x = layers.Conv2D(filters=32, kernel_size=3,activation='relu',padding = 'same')(x) # 필터 32
# x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3,activation='relu',padding = 'same')(x) # 필터 64
# x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3,activation='relu',padding = 'same')(x) # 필터 64
# x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3,activation='relu',padding = 'same')(x) # 필터 128
# x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=256, kernel_size=3,activation='relu',padding = 'same')(x) # 필터 128
# x = layers.Flatten()(x)   # 드롭아웃 설정
#
# outputs = layers.Dense(3, activation = "softmax")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# # 모델 컴파일
# model.compile(optimizer='Adam',
#               loss='sparse_categorical_crossentropy', # 다중 클래스 분류 손실 함수 사용
#               metrics=['accuracy'])     # 평가 지표로 정확도 사용
#
#
# seed=np.random.randint(0, 1000)
# # train 데이터의 일부를 학습 데이터로 사용
# train_dataset = image_dataset_from_directory(
#     train_dir,                          # 학습 데이터 경로
#     image_size=(224, 224),              # 이미지 크기 조정
#     batch_size=batch,                   # 배치 크기
#     validation_split=split,               # 검증 데이터 비율
#     subset="training",                  # 학습용 데이터로 지정
#     seed=seed                           # 데이터 셔플을 위한 시드
# )
#
# # train 데이터의 일부를 검증 데이터로 사용
# validation_dataset = image_dataset_from_directory(
#     train_dir,                          # 학습 데이터 경로
#     image_size=(224, 224),              # 이미지 크기
#     batch_size=batch,                   # 배치 크기
#     validation_split=split,               # 검증 데이터 비율
#     subset="validation",                # 검증용 데이터로 지정
#     seed=seed                           # 데이터 셔플을 위한 시드
# )
#
# # 테스트 데이터
# test_dataset = image_dataset_from_directory(
#     test_dir,                           # 테스트 데이터 경로
#     image_size=(224, 224),              # 이미지 크기 조정
#     batch_size=batch,                   # 배치 크기
# )
#
# # 데이터 증강
# data_augmentation = keras.Sequential([
#     layers.RandomFlip("horizontal"),  # 좌우 반전
#     layers.RandomRotation(0.1),       # 10% 회전
#     layers.RandomZoom(0.2),           # 20% 확대/축소
# ])
#
# # 데이터셋에 증강 적용
# train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
#
# # Callback
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath="convnet_from_scratch.keras",     # HDF5 오류로 SavedModel 형식으로 지정
#         save_best_only=True,
#         monitor="val_loss"
#     )
# ]
#
# # 모델 학습
# history = model.fit(
#     train_dataset,                      # 학습 데이터셋 지정
#     epochs=epoch,                       # 에폭 설정
#     validation_data=validation_dataset, # 검증 데이터셋 지정
#     callbacks=callbacks)                # callback 설정
#
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--', marker='s')
# plt.title('Training and Validation Accuracy', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.show()
#
# # Training and Validation Loss Plot
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o', color='red')
# plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--', marker='s', color='orange')
# plt.title('Training and Validation Loss', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.show()
#
# test_model = keras.models.load_model("convnet_from_scratch.keras")
# test_loss, test_acc = test_model.evaluate(test_dataset)
# print(f"정확도: {test_acc:.3f}")
