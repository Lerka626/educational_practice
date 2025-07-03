# Task 1.1 Расширение линейной регрессии - Terminal:
True
Размер датасета: 200
Количество батчей: 7
Epoch 10: loss=0.0736
Epoch 20: loss=0.0220
Epoch 30: loss=0.0128
Epoch 40: loss=0.0114
2025-07-02 22:38:17,777 [INFO] Ранняя остановка сработала на эпохе: 49


# Task 1.2 Расширение логистической регрессии - Terminal:
Размер датасета: 300
Количество батчей: 10
Epoch 10: loss=0.5242, acc=0.9433
Epoch 20: loss=0.4027, acc=0.9600
Epoch 30: loss=0.3356, acc=0.9600
Epoch 40: loss=0.2995, acc=0.9700
Epoch 50: loss=0.2741, acc=0.9800
Epoch 60: loss=0.2529, acc=0.9833
Epoch 70: loss=0.2358, acc=0.9767
Epoch 80: loss=0.2188, acc=0.9833
Epoch 90: loss=0.2059, acc=0.9867
Epoch 100: loss=0.1986, acc=0.9833
2025-07-03 08:00:27,552 [INFO] Метрики по эпохам:
2025-07-03 08:00:27,553 [INFO] Accuracy:  0.9833
2025-07-03 08:00:27,560 [INFO] Precision: 0.9831
2025-07-03 08:00:27,567 [INFO] Recall:    0.9835
2025-07-03 08:00:27,574 [INFO] F1-score:  0.9833
c:\Users\Lerik\OneDrive\Desktop\all_practices\env\Lib\site-packages\sklearn\metrics\_ranking.py:424: UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
  warnings.warn(
2025-07-03 08:00:27,581 [INFO] ROC-AUC:   nan
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK


# Task 2.1 Кастомный Dataset класс - Terminal:
2025-07-03 08:36:05,643 [INFO] Загрузка данных из student_habits_performance.csv
2025-07-03 08:36:05,653 [INFO] Применение пайплайна для обработки признаков
Размер: 1000
X shape: torch.Size([25]), Y: 56
2025-07-03 08:36:05,710 [INFO] Визуализация распределений признаков
2025-07-03 08:40:23,249 [INFO] Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2025-07-03 08:40:23,263 [INFO] Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
2025-07-03 08:41:03,812 [INFO] Загрузка данных из temp_student_test.csv
2025-07-03 08:41:03,833 [INFO] Применение пайплайна для обработки признаков
.
----------------------------------------------------------------------
Ran 1 test in 0.092s

OK


# Task 2.2 Эксперименты с различными датасетами - Terminal:
--LinReg
В терминале:
Epoch 10: loss=0.0537
Epoch 20: loss=0.0536
Точность (accuracy): 0.9478

# --LogReg
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
0   63    1   3       145   233    1        0      150      0      2.3      0   0     1       1
1   37    1   2       130   250    0        1      187      0      3.5      0   0     2       1
2   41    0   1       130   204    0        0      172      0      1.4      2   0     2       1
3   56    1   1       120   236    0        1      178      0      0.8      2   0     2       1
4   57    0   0       120   354    0        1      163      1      0.6      2   0     2       1
target
1    165
0    138
Name: count, dtype: int64
Классов: 2
Epoch 10: loss=0.3659, acc=0.8471
Epoch 20: loss=0.3462, acc=0.8595
Epoch 30: loss=0.3428, acc=0.8595
Epoch 40: loss=0.3583, acc=0.8595
Epoch 50: loss=0.3447, acc=0.8678
Epoch 60: loss=0.3427, acc=0.8636
Epoch 70: loss=0.3430, acc=0.8719
Epoch 80: loss=0.3650, acc=0.8636
Epoch 90: loss=0.3509, acc=0.8678
Epoch 100: loss=0.3531, acc=0.8719



# Task 3.1 Исследование гиперпараметров - Terminal:
Epoch 10: loss=0.8028, acc=0.5331
Epoch 20: loss=0.6552, acc=0.6364
Epoch 30: loss=0.5910, acc=0.7273
Epoch 40: loss=0.5572, acc=0.7769
Epoch 50: loss=0.5309, acc=0.8099
2025-07-03 09:40:36,949 [INFO] Optimizer: SGD, LR: 0.001, Batch size: 16
2025-07-03 09:40:36,949 [INFO] Метрики по эпохам:
2025-07-03 09:40:36,949 [INFO] Accuracy:  0.7541
2025-07-03 09:40:36,955 [INFO] Precision: 0.7527
2025-07-03 09:40:36,960 [INFO] Recall:    0.7538
2025-07-03 09:40:36,963 [INFO] F1-score:  0.7530
2025-07-03 09:40:36,974 [INFO] ROC-AUC:   0.7576
Epoch 10: loss=0.6431, acc=0.6364
Epoch 20: loss=0.6115, acc=0.6942
Epoch 30: loss=0.5789, acc=0.7107
Epoch 40: loss=0.5503, acc=0.7190
Epoch 50: loss=0.5381, acc=0.7397
2025-07-03 09:40:37,561 [INFO] Optimizer: SGD, LR: 0.001, Batch size: 32
2025-07-03 09:40:37,561 [INFO] Метрики по эпохам:
2025-07-03 09:40:37,562 [INFO] Accuracy:  0.6885
2025-07-03 09:40:37,566 [INFO] Precision: 0.6932
2025-07-03 09:40:37,569 [INFO] Recall:    0.6932
2025-07-03 09:40:37,572 [INFO] F1-score:  0.6885
2025-07-03 09:40:37,576 [INFO] ROC-AUC:   0.7900
Epoch 10: loss=0.7410, acc=0.5289
Epoch 20: loss=0.7192, acc=0.5620
Epoch 30: loss=0.6922, acc=0.5868
Epoch 40: loss=0.6725, acc=0.5992
Epoch 50: loss=0.6515, acc=0.5868
2025-07-03 09:40:37,978 [INFO] Optimizer: SGD, LR: 0.001, Batch size: 64
2025-07-03 09:40:37,978 [INFO] Метрики по эпохам:
2025-07-03 09:40:37,979 [INFO] Accuracy:  0.6230
2025-07-03 09:40:37,982 [INFO] Precision: 0.6380
2025-07-03 09:40:37,986 [INFO] Recall:    0.6326
2025-07-03 09:40:37,989 [INFO] F1-score:  0.6213
2025-07-03 09:40:37,992 [INFO] ROC-AUC:   0.6180
Epoch 10: loss=0.3914, acc=0.8264
Epoch 20: loss=0.3824, acc=0.8512
Epoch 30: loss=0.3502, acc=0.8554
Epoch 40: loss=0.3926, acc=0.8512
Epoch 50: loss=0.3388, acc=0.8595
2025-07-03 09:40:38,865 [INFO] Optimizer: SGD, LR: 0.01, Batch size: 16
2025-07-03 09:40:38,865 [INFO] Метрики по эпохам:
2025-07-03 09:40:38,866 [INFO] Accuracy:  0.7869
2025-07-03 09:40:38,869 [INFO] Precision: 0.8036
2025-07-03 09:40:38,871 [INFO] Recall:    0.7760
2025-07-03 09:40:38,875 [INFO] F1-score:  0.7783
2025-07-03 09:40:38,878 [INFO] ROC-AUC:   0.8777
Epoch 10: loss=0.4905, acc=0.7893
Epoch 20: loss=0.4164, acc=0.8182
Epoch 30: loss=0.3829, acc=0.8223
Epoch 40: loss=0.3777, acc=0.8471
Epoch 50: loss=0.3734, acc=0.8512
2025-07-03 09:40:39,541 [INFO] Optimizer: SGD, LR: 0.01, Batch size: 32
2025-07-03 09:40:39,541 [INFO] Метрики по эпохам:
2025-07-03 09:40:39,541 [INFO] Accuracy:  0.8197
2025-07-03 09:40:39,544 [INFO] Precision: 0.8399
2025-07-03 09:40:39,547 [INFO] Recall:    0.8090
2025-07-03 09:40:39,550 [INFO] F1-score:  0.8124
2025-07-03 09:40:39,554 [INFO] ROC-AUC:   0.8939
Epoch 10: loss=0.5246, acc=0.7686
Epoch 20: loss=0.4640, acc=0.8140
Epoch 30: loss=0.4321, acc=0.8140
Epoch 40: loss=0.4158, acc=0.8264
Epoch 50: loss=0.3993, acc=0.8223
2025-07-03 09:40:39,949 [INFO] Optimizer: SGD, LR: 0.01, Batch size: 64
2025-07-03 09:40:39,949 [INFO] Метрики по эпохам:
2025-07-03 09:40:39,950 [INFO] Accuracy:  0.8525
2025-07-03 09:40:39,953 [INFO] Precision: 0.8762
2025-07-03 09:40:39,957 [INFO] Recall:    0.8420
2025-07-03 09:40:39,960 [INFO] F1-score:  0.8465
2025-07-03 09:40:39,963 [INFO] ROC-AUC:   0.8983
Epoch 10: loss=0.3747, acc=0.8595
Epoch 20: loss=0.3323, acc=0.8388
Epoch 30: loss=0.3351, acc=0.8347
Epoch 40: loss=0.3768, acc=0.8347
Epoch 50: loss=0.3521, acc=0.8512
2025-07-03 09:40:40,836 [INFO] Optimizer: SGD, LR: 0.1, Batch size: 16
2025-07-03 09:40:40,836 [INFO] Метрики по эпохам:
2025-07-03 09:40:40,836 [INFO] Accuracy:  0.8033
2025-07-03 09:40:40,839 [INFO] Precision: 0.8164
2025-07-03 09:40:40,842 [INFO] Recall:    0.7938
2025-07-03 09:40:40,845 [INFO] F1-score:  0.7967
2025-07-03 09:40:40,848 [INFO] ROC-AUC:   0.8723
Epoch 10: loss=0.3466, acc=0.8471
Epoch 20: loss=0.3361, acc=0.8388
Epoch 30: loss=0.3423, acc=0.8388
Epoch 40: loss=0.3467, acc=0.8430
Epoch 50: loss=0.3405, acc=0.8430
2025-07-03 09:40:41,421 [INFO] Optimizer: SGD, LR: 0.1, Batch size: 32
2025-07-03 09:40:41,421 [INFO] Метрики по эпохам:
2025-07-03 09:40:41,422 [INFO] Accuracy:  0.8033
2025-07-03 09:40:41,426 [INFO] Precision: 0.8164
2025-07-03 09:40:41,429 [INFO] Recall:    0.7938
2025-07-03 09:40:41,432 [INFO] F1-score:  0.7967
2025-07-03 09:40:41,435 [INFO] ROC-AUC:   0.8680
Epoch 10: loss=0.3628, acc=0.8430
Epoch 20: loss=0.3459, acc=0.8554
Epoch 30: loss=0.3398, acc=0.8471
Epoch 40: loss=0.3451, acc=0.8430
Epoch 50: loss=0.3443, acc=0.8388
2025-07-03 09:40:41,835 [INFO] Optimizer: SGD, LR: 0.1, Batch size: 64
2025-07-03 09:40:41,835 [INFO] Метрики по эпохам:
2025-07-03 09:40:41,836 [INFO] Accuracy:  0.8033
2025-07-03 09:40:41,839 [INFO] Precision: 0.8164
2025-07-03 09:40:41,843 [INFO] Recall:    0.7938
2025-07-03 09:40:41,846 [INFO] F1-score:  0.7967
2025-07-03 09:40:41,849 [INFO] ROC-AUC:   0.8701
Epoch 10: loss=0.4379, acc=0.8140
Epoch 20: loss=0.3773, acc=0.8512
Epoch 30: loss=0.3734, acc=0.8554
Epoch 40: loss=0.3387, acc=0.8471
Epoch 50: loss=0.3333, acc=0.8512
2025-07-03 09:40:42,947 [INFO] Optimizer: Adam, LR: 0.001, Batch size: 16
2025-07-03 09:40:42,947 [INFO] Метрики по эпохам:
2025-07-03 09:40:42,947 [INFO] Accuracy:  0.7869
2025-07-03 09:40:42,949 [INFO] Precision: 0.8036
2025-07-03 09:40:42,953 [INFO] Recall:    0.7760
2025-07-03 09:40:42,956 [INFO] F1-score:  0.7783
2025-07-03 09:40:42,959 [INFO] ROC-AUC:   0.8864
Epoch 10: loss=0.5194, acc=0.7521
Epoch 20: loss=0.4372, acc=0.7975
Epoch 30: loss=0.4021, acc=0.8306
Epoch 40: loss=0.3767, acc=0.8347
Epoch 50: loss=0.3728, acc=0.8347
2025-07-03 09:40:43,627 [INFO] Optimizer: Adam, LR: 0.001, Batch size: 32
2025-07-03 09:40:43,627 [INFO] Метрики по эпохам:
2025-07-03 09:40:43,628 [INFO] Accuracy:  0.8033
2025-07-03 09:40:43,631 [INFO] Precision: 0.8164
2025-07-03 09:40:43,635 [INFO] Recall:    0.7938
2025-07-03 09:40:43,639 [INFO] F1-score:  0.7967
2025-07-03 09:40:43,642 [INFO] ROC-AUC:   0.8885
Epoch 10: loss=0.6991, acc=0.6033
Epoch 20: loss=0.5927, acc=0.6983
Epoch 30: loss=0.5280, acc=0.7397
Epoch 40: loss=0.4811, acc=0.7479
Epoch 50: loss=0.4525, acc=0.7975
2025-07-03 09:40:44,092 [INFO] Optimizer: Adam, LR: 0.001, Batch size: 64
2025-07-03 09:40:44,092 [INFO] Метрики по эпохам:
2025-07-03 09:40:44,092 [INFO] Accuracy:  0.7869
2025-07-03 09:40:44,095 [INFO] Precision: 0.8036
2025-07-03 09:40:44,098 [INFO] Recall:    0.7760
2025-07-03 09:40:44,101 [INFO] F1-score:  0.7783
2025-07-03 09:40:44,105 [INFO] ROC-AUC:   0.8517
Epoch 10: loss=0.3367, acc=0.8388
Epoch 20: loss=0.3324, acc=0.8554
Epoch 30: loss=0.3296, acc=0.8471
Epoch 40: loss=0.3327, acc=0.8471
Epoch 50: loss=0.3337, acc=0.8347
2025-07-03 09:40:45,189 [INFO] Optimizer: Adam, LR: 0.01, Batch size: 16
2025-07-03 09:40:45,191 [INFO] Метрики по эпохам:
2025-07-03 09:40:45,191 [INFO] Accuracy:  0.8033
2025-07-03 09:40:45,194 [INFO] Precision: 0.8164
2025-07-03 09:40:45,197 [INFO] Recall:    0.7938
2025-07-03 09:40:45,200 [INFO] F1-score:  0.7967
2025-07-03 09:40:45,204 [INFO] ROC-AUC:   0.8680
Epoch 10: loss=0.3353, acc=0.8388
Epoch 20: loss=0.3446, acc=0.8430
Epoch 30: loss=0.3350, acc=0.8430
Epoch 40: loss=0.3605, acc=0.8430
Epoch 50: loss=0.3587, acc=0.8471
2025-07-03 09:40:45,877 [INFO] Optimizer: Adam, LR: 0.01, Batch size: 32
2025-07-03 09:40:45,877 [INFO] Метрики по эпохам:
2025-07-03 09:40:45,878 [INFO] Accuracy:  0.8033
2025-07-03 09:40:45,882 [INFO] Precision: 0.8164
2025-07-03 09:40:45,884 [INFO] Recall:    0.7938
2025-07-03 09:40:45,887 [INFO] F1-score:  0.7967
2025-07-03 09:40:45,891 [INFO] ROC-AUC:   0.8658
Epoch 10: loss=0.3533, acc=0.8471
Epoch 20: loss=0.3412, acc=0.8430
Epoch 30: loss=0.3378, acc=0.8430
Epoch 40: loss=0.3377, acc=0.8471
Epoch 50: loss=0.3409, acc=0.8471
2025-07-03 09:40:46,348 [INFO] Optimizer: Adam, LR: 0.01, Batch size: 64
2025-07-03 09:40:46,348 [INFO] Метрики по эпохам:
2025-07-03 09:40:46,349 [INFO] Accuracy:  0.8033
2025-07-03 09:40:46,352 [INFO] Precision: 0.8164
2025-07-03 09:40:46,354 [INFO] Recall:    0.7938
2025-07-03 09:40:46,357 [INFO] F1-score:  0.7967
2025-07-03 09:40:46,361 [INFO] ROC-AUC:   0.8680
Epoch 10: loss=0.4168, acc=0.7893
Epoch 20: loss=0.4142, acc=0.8223
Epoch 30: loss=0.3623, acc=0.8388
Epoch 40: loss=0.4966, acc=0.8347
Epoch 50: loss=0.3691, acc=0.8471
2025-07-03 09:40:47,447 [INFO] Optimizer: Adam, LR: 0.1, Batch size: 16
2025-07-03 09:40:47,448 [INFO] Метрики по эпохам:
2025-07-03 09:40:47,448 [INFO] Accuracy:  0.8033
2025-07-03 09:40:47,456 [INFO] Precision: 0.8164
2025-07-03 09:40:47,459 [INFO] Recall:    0.7938
2025-07-03 09:40:47,462 [INFO] F1-score:  0.7967
2025-07-03 09:40:47,465 [INFO] ROC-AUC:   0.8961
Epoch 10: loss=0.3661, acc=0.8388
Epoch 20: loss=0.3797, acc=0.8554
Epoch 30: loss=0.3646, acc=0.8388
Epoch 40: loss=0.3630, acc=0.8306
Epoch 50: loss=0.3533, acc=0.8430
2025-07-03 09:40:48,131 [INFO] Optimizer: Adam, LR: 0.1, Batch size: 32
2025-07-03 09:40:48,131 [INFO] Метрики по эпохам:
2025-07-03 09:40:48,132 [INFO] Accuracy:  0.8033
2025-07-03 09:40:48,135 [INFO] Precision: 0.8164
2025-07-03 09:40:48,137 [INFO] Recall:    0.7938
2025-07-03 09:40:48,140 [INFO] F1-score:  0.7967
2025-07-03 09:40:48,144 [INFO] ROC-AUC:   0.8561
Epoch 10: loss=0.3473, acc=0.8430
Epoch 20: loss=0.3522, acc=0.8471
Epoch 30: loss=0.3478, acc=0.8512
Epoch 40: loss=0.3439, acc=0.8471
Epoch 50: loss=0.3488, acc=0.8430
2025-07-03 09:40:48,603 [INFO] Optimizer: Adam, LR: 0.1, Batch size: 64
2025-07-03 09:40:48,603 [INFO] Метрики по эпохам:
2025-07-03 09:40:48,603 [INFO] Accuracy:  0.8033
2025-07-03 09:40:48,606 [INFO] Precision: 0.8164
2025-07-03 09:40:48,609 [INFO] Recall:    0.7938
2025-07-03 09:40:48,613 [INFO] F1-score:  0.7967
2025-07-03 09:40:48,616 [INFO] ROC-AUC:   0.8734
Epoch 10: loss=0.4205, acc=0.8306
Epoch 20: loss=0.3609, acc=0.8512
Epoch 30: loss=0.3455, acc=0.8471
Epoch 40: loss=0.3654, acc=0.8430
Epoch 50: loss=0.3385, acc=0.8471
2025-07-03 09:40:49,737 [INFO] Optimizer: RMSprop, LR: 0.001, Batch size: 16
2025-07-03 09:40:49,738 [INFO] Метрики по эпохам:
2025-07-03 09:40:49,738 [INFO] Accuracy:  0.7869
2025-07-03 09:40:49,742 [INFO] Precision: 0.8036
2025-07-03 09:40:49,745 [INFO] Recall:    0.7760
2025-07-03 09:40:49,750 [INFO] F1-score:  0.7783
2025-07-03 09:40:49,753 [INFO] ROC-AUC:   0.8907
Epoch 10: loss=0.4890, acc=0.7810
Epoch 20: loss=0.4225, acc=0.8140
Epoch 30: loss=0.4002, acc=0.8264
Epoch 40: loss=0.3865, acc=0.8471
Epoch 50: loss=0.3771, acc=0.8388
2025-07-03 09:40:50,387 [INFO] Optimizer: RMSprop, LR: 0.001, Batch size: 32
2025-07-03 09:40:50,387 [INFO] Метрики по эпохам:
2025-07-03 09:40:50,387 [INFO] Accuracy:  0.8033
2025-07-03 09:40:50,390 [INFO] Precision: 0.8164
2025-07-03 09:40:50,393 [INFO] Recall:    0.7938
2025-07-03 09:40:50,399 [INFO] F1-score:  0.7967
2025-07-03 09:40:50,403 [INFO] ROC-AUC:   0.8994
Epoch 10: loss=0.5423, acc=0.7314
Epoch 20: loss=0.4765, acc=0.7769
Epoch 30: loss=0.4375, acc=0.8017
Epoch 40: loss=0.4242, acc=0.8140
Epoch 50: loss=0.4066, acc=0.8306
2025-07-03 09:40:50,846 [INFO] Optimizer: RMSprop, LR: 0.001, Batch size: 64
2025-07-03 09:40:50,847 [INFO] Метрики по эпохам:
2025-07-03 09:40:50,847 [INFO] Accuracy:  0.8525
2025-07-03 09:40:50,851 [INFO] Precision: 0.8524
2025-07-03 09:40:50,855 [INFO] Recall:    0.8501
2025-07-03 09:40:50,858 [INFO] F1-score:  0.8510
2025-07-03 09:40:50,862 [INFO] ROC-AUC:   0.8842
Epoch 10: loss=0.3526, acc=0.8347
Epoch 20: loss=0.3324, acc=0.8388
Epoch 30: loss=0.4428, acc=0.8430
Epoch 40: loss=0.3957, acc=0.8388
Epoch 50: loss=0.3379, acc=0.8388
2025-07-03 09:40:51,916 [INFO] Optimizer: RMSprop, LR: 0.01, Batch size: 16
2025-07-03 09:40:51,917 [INFO] Метрики по эпохам:
2025-07-03 09:40:51,917 [INFO] Accuracy:  0.8033
2025-07-03 09:40:51,921 [INFO] Precision: 0.8164
2025-07-03 09:40:51,925 [INFO] Recall:    0.7938
2025-07-03 09:40:51,928 [INFO] F1-score:  0.7967
2025-07-03 09:40:51,931 [INFO] ROC-AUC:   0.8755
Epoch 10: loss=0.3424, acc=0.8388
Epoch 20: loss=0.3508, acc=0.8388
Epoch 30: loss=0.3383, acc=0.8347
Epoch 40: loss=0.3538, acc=0.8471
Epoch 50: loss=0.3578, acc=0.8347
2025-07-03 09:40:52,577 [INFO] Optimizer: RMSprop, LR: 0.01, Batch size: 32
2025-07-03 09:40:52,577 [INFO] Метрики по эпохам:
2025-07-03 09:40:52,578 [INFO] Accuracy:  0.8033
2025-07-03 09:40:52,581 [INFO] Precision: 0.8164
2025-07-03 09:40:52,584 [INFO] Recall:    0.7938
2025-07-03 09:40:52,587 [INFO] F1-score:  0.7967
2025-07-03 09:40:52,591 [INFO] ROC-AUC:   0.8680
Epoch 10: loss=0.3466, acc=0.8388
Epoch 20: loss=0.3416, acc=0.8430
Epoch 30: loss=0.3452, acc=0.8471
Epoch 40: loss=0.3494, acc=0.8430
Epoch 50: loss=0.3409, acc=0.8430
2025-07-03 09:40:53,032 [INFO] Optimizer: RMSprop, LR: 0.01, Batch size: 64
2025-07-03 09:40:53,033 [INFO] Метрики по эпохам:
2025-07-03 09:40:53,033 [INFO] Accuracy:  0.8033
2025-07-03 09:40:53,036 [INFO] Precision: 0.8164
2025-07-03 09:40:53,039 [INFO] Recall:    0.7938
2025-07-03 09:40:53,041 [INFO] F1-score:  0.7967
2025-07-03 09:40:53,045 [INFO] ROC-AUC:   0.8636
Epoch 10: loss=0.4414, acc=0.8017
Epoch 20: loss=0.3875, acc=0.8264
Epoch 30: loss=0.3947, acc=0.8306
Epoch 40: loss=0.4861, acc=0.8264
Epoch 50: loss=0.4117, acc=0.8471
2025-07-03 09:40:54,073 [INFO] Optimizer: RMSprop, LR: 0.1, Batch size: 16
2025-07-03 09:40:54,074 [INFO] Метрики по эпохам:
2025-07-03 09:40:54,074 [INFO] Accuracy:  0.7213
2025-07-03 09:40:54,079 [INFO] Precision: 0.7198
2025-07-03 09:40:54,082 [INFO] Recall:    0.7208
2025-07-03 09:40:54,085 [INFO] F1-score:  0.7201
2025-07-03 09:40:54,089 [INFO] ROC-AUC:   0.7749
Epoch 10: loss=0.4024, acc=0.8140
Epoch 20: loss=0.4216, acc=0.8099
Epoch 30: loss=0.3860, acc=0.8306
Epoch 40: loss=0.3857, acc=0.8264
Epoch 50: loss=0.3942, acc=0.8347
2025-07-03 09:40:54,752 [INFO] Optimizer: RMSprop, LR: 0.1, Batch size: 32
2025-07-03 09:40:54,752 [INFO] Метрики по эпохам:
2025-07-03 09:40:54,753 [INFO] Accuracy:  0.7869
2025-07-03 09:40:54,756 [INFO] Precision: 0.7946
2025-07-03 09:40:54,760 [INFO] Recall:    0.7787
2025-07-03 09:40:54,764 [INFO] F1-score:  0.7810
2025-07-03 09:40:54,766 [INFO] ROC-AUC:   0.8323
Epoch 10: loss=0.4132, acc=0.8223
Epoch 20: loss=0.4117, acc=0.8140
Epoch 30: loss=0.4108, acc=0.8140
Epoch 40: loss=0.3802, acc=0.8264
Epoch 50: loss=0.3819, acc=0.8264
2025-07-03 09:40:55,206 [INFO] Optimizer: RMSprop, LR: 0.1, Batch size: 64
2025-07-03 09:40:55,208 [INFO] Метрики по эпохам:
2025-07-03 09:40:55,208 [INFO] Accuracy:  0.8033
2025-07-03 09:40:55,211 [INFO] Precision: 0.8019
2025-07-03 09:40:55,214 [INFO] Recall:    0.8019
2025-07-03 09:40:55,217 [INFO] F1-score:  0.8019
2025-07-03 09:40:55,220 [INFO] ROC-AUC:   0.8777
   optimizer  learning_rate  batch_size  accuracy_val
0        SGD          0.001          16      0.754098
1        SGD          0.001          32      0.688525
2        SGD          0.001          64      0.622951
3        SGD          0.010          16      0.786885
4        SGD          0.010          32      0.819672
5        SGD          0.010          64      0.852459
6        SGD          0.100          16      0.803279
7        SGD          0.100          32      0.803279
8        SGD          0.100          64      0.803279
9       Adam          0.001          16      0.786885
10      Adam          0.001          32      0.803279
11      Adam          0.001          64      0.786885
12      Adam          0.010          16      0.803279
13      Adam          0.010          32      0.803279
14      Adam          0.010          64      0.803279
15      Adam          0.100          16      0.803279
16      Adam          0.100          32      0.803279
17      Adam          0.100          64      0.803279
18   RMSprop          0.001          16      0.786885
19   RMSprop          0.001          32      0.803279
20   RMSprop          0.001          64      0.852459
21   RMSprop          0.010          16      0.803279
22   RMSprop          0.010          32      0.803279
23   RMSprop          0.010          64      0.803279
24   RMSprop          0.100          16      0.721311
25   RMSprop          0.100          32      0.786885
26   RMSprop          0.100          64      0.803279


# Task 3.2 Feature Engineering - Terminal:

Epoch 10: loss=0.6281, acc=0.6570
Epoch 20: loss=0.5587, acc=0.7107
Epoch 30: loss=0.5310, acc=0.7645
Epoch 40: loss=0.5306, acc=0.7934
Epoch 50: loss=0.4997, acc=0.8140
2025-07-03 09:56:40,566 [INFO] Optimizer: SGD, LR: 0.001, Batch size: 16
2025-07-03 09:56:40,566 [INFO] Метрики по эпохам:
2025-07-03 09:56:40,567 [INFO] Accuracy:  0.6721
2025-07-03 09:56:40,571 [INFO] Precision: 0.6699
2025-07-03 09:56:40,574 [INFO] Recall:    0.6699
2025-07-03 09:56:40,577 [INFO] F1-score:  0.6699
2025-07-03 09:56:40,581 [INFO] ROC-AUC:   0.7803
Epoch 10: loss=0.8142, acc=0.4091
Epoch 20: loss=0.7413, acc=0.4959
Epoch 30: loss=0.6870, acc=0.5620
Epoch 40: loss=0.6388, acc=0.6322
Epoch 50: loss=0.5950, acc=0.6860
2025-07-03 09:56:41,189 [INFO] Optimizer: SGD, LR: 0.001, Batch size: 32
2025-07-03 09:56:41,189 [INFO] Метрики по эпохам:
2025-07-03 09:56:41,190 [INFO] Accuracy:  0.7213
2025-07-03 09:56:41,194 [INFO] Precision: 0.7310
2025-07-03 09:56:41,198 [INFO] Recall:    0.7100
2025-07-03 09:56:41,202 [INFO] F1-score:  0.7101
2025-07-03 09:56:41,205 [INFO] ROC-AUC:   0.7381
Epoch 10: loss=0.6378, acc=0.6529
Epoch 20: loss=0.6298, acc=0.6736
Epoch 30: loss=0.6048, acc=0.6901
Epoch 40: loss=0.5926, acc=0.6983
Epoch 50: loss=0.5799, acc=0.7149
2025-07-03 09:56:41,742 [INFO] Optimizer: SGD, LR: 0.001, Batch size: 64
2025-07-03 09:56:41,742 [INFO] Метрики по эпохам:
2025-07-03 09:56:41,743 [INFO] Accuracy:  0.6230
2025-07-03 09:56:41,746 [INFO] Precision: 0.6198
2025-07-03 09:56:41,750 [INFO] Recall:    0.6190
2025-07-03 09:56:41,753 [INFO] F1-score:  0.6193
2025-07-03 09:56:41,758 [INFO] ROC-AUC:   0.6851
Epoch 10: loss=0.4391, acc=0.7975
Epoch 20: loss=0.3625, acc=0.8388
Epoch 30: loss=0.3515, acc=0.8471
Epoch 40: loss=0.3501, acc=0.8554
Epoch 50: loss=0.3521, acc=0.8554
2025-07-03 09:56:42,761 [INFO] Optimizer: SGD, LR: 0.01, Batch size: 16
2025-07-03 09:56:42,761 [INFO] Метрики по эпохам:
2025-07-03 09:56:42,762 [INFO] Accuracy:  0.7869
2025-07-03 09:56:42,765 [INFO] Precision: 0.8036
2025-07-03 09:56:42,769 [INFO] Recall:    0.7760
2025-07-03 09:56:42,772 [INFO] F1-score:  0.7783
2025-07-03 09:56:42,775 [INFO] ROC-AUC:   0.8788
Epoch 10: loss=0.4673, acc=0.8017
Epoch 20: loss=0.4075, acc=0.8388
Epoch 30: loss=0.3801, acc=0.8388
Epoch 40: loss=0.3668, acc=0.8471
Epoch 50: loss=0.3592, acc=0.8471
2025-07-03 09:56:43,365 [INFO] Optimizer: SGD, LR: 0.01, Batch size: 32
2025-07-03 09:56:43,365 [INFO] Метрики по эпохам:
2025-07-03 09:56:43,366 [INFO] Accuracy:  0.7869
2025-07-03 09:56:43,370 [INFO] Precision: 0.8036
2025-07-03 09:56:43,373 [INFO] Recall:    0.7760
2025-07-03 09:56:43,377 [INFO] F1-score:  0.7783
2025-07-03 09:56:43,380 [INFO] ROC-AUC:   0.8929
Epoch 10: loss=0.5420, acc=0.8017
Epoch 20: loss=0.4675, acc=0.8223
Epoch 30: loss=0.4296, acc=0.8347
Epoch 40: loss=0.4030, acc=0.8347
Epoch 50: loss=0.3885, acc=0.8388
2025-07-03 09:56:43,840 [INFO] Optimizer: SGD, LR: 0.01, Batch size: 64
2025-07-03 09:56:43,841 [INFO] Метрики по эпохам:
2025-07-03 09:56:43,841 [INFO] Accuracy:  0.7705
2025-07-03 09:56:43,845 [INFO] Precision: 0.7742
2025-07-03 09:56:43,849 [INFO] Recall:    0.7635
2025-07-03 09:56:43,853 [INFO] F1-score:  0.7654
2025-07-03 09:56:43,856 [INFO] ROC-AUC:   0.8777
Epoch 10: loss=0.3750, acc=0.8388
Epoch 20: loss=0.3310, acc=0.8430
Epoch 30: loss=0.3307, acc=0.8388
Epoch 40: loss=0.3520, acc=0.8471
Epoch 50: loss=0.3303, acc=0.8306
2025-07-03 09:56:44,872 [INFO] Optimizer: SGD, LR: 0.1, Batch size: 16
2025-07-03 09:56:44,872 [INFO] Метрики по эпохам:
2025-07-03 09:56:44,872 [INFO] Accuracy:  0.8033
2025-07-03 09:56:44,876 [INFO] Precision: 0.8164
2025-07-03 09:56:44,878 [INFO] Recall:    0.7938
2025-07-03 09:56:44,883 [INFO] F1-score:  0.7967
2025-07-03 09:56:44,887 [INFO] ROC-AUC:   0.8615
Epoch 10: loss=0.3383, acc=0.8388
Epoch 20: loss=0.3730, acc=0.8512
Epoch 30: loss=0.3448, acc=0.8471
Epoch 40: loss=0.3366, acc=0.8430
Epoch 50: loss=0.3391, acc=0.8347
2025-07-03 09:56:45,523 [INFO] Optimizer: SGD, LR: 0.1, Batch size: 32
2025-07-03 09:56:45,523 [INFO] Метрики по эпохам:
2025-07-03 09:56:45,524 [INFO] Accuracy:  0.8033
2025-07-03 09:56:45,527 [INFO] Precision: 0.8164
2025-07-03 09:56:45,531 [INFO] Recall:    0.7938
2025-07-03 09:56:45,535 [INFO] F1-score:  0.7967
2025-07-03 09:56:45,538 [INFO] ROC-AUC:   0.8701
Epoch 10: loss=0.3572, acc=0.8471
Epoch 20: loss=0.3474, acc=0.8554
Epoch 30: loss=0.3366, acc=0.8512
Epoch 40: loss=0.3375, acc=0.8471
Epoch 50: loss=0.3411, acc=0.8430
2025-07-03 09:56:45,982 [INFO] Optimizer: SGD, LR: 0.1, Batch size: 64
2025-07-03 09:56:45,982 [INFO] Метрики по эпохам:
2025-07-03 09:56:45,983 [INFO] Accuracy:  0.8033
2025-07-03 09:56:45,986 [INFO] Precision: 0.8164
2025-07-03 09:56:45,991 [INFO] Recall:    0.7938
2025-07-03 09:56:45,994 [INFO] F1-score:  0.7967
2025-07-03 09:56:45,998 [INFO] ROC-AUC:   0.8712
Epoch 10: loss=0.5126, acc=0.8017
Epoch 20: loss=0.4064, acc=0.8223
Epoch 30: loss=0.3659, acc=0.8388
Epoch 40: loss=0.3568, acc=0.8430
Epoch 50: loss=0.3399, acc=0.8512
2025-07-03 09:56:47,378 [INFO] Optimizer: Adam, LR: 0.001, Batch size: 16
2025-07-03 09:56:47,379 [INFO] Метрики по эпохам:
2025-07-03 09:56:47,379 [INFO] Accuracy:  0.7869
2025-07-03 09:56:47,383 [INFO] Precision: 0.8036
2025-07-03 09:56:47,386 [INFO] Recall:    0.7760
2025-07-03 09:56:47,389 [INFO] F1-score:  0.7783
2025-07-03 09:56:47,392 [INFO] ROC-AUC:   0.8983
Epoch 10: loss=0.5276, acc=0.7438
Epoch 20: loss=0.4373, acc=0.8099
Epoch 30: loss=0.4036, acc=0.8264
Epoch 40: loss=0.3777, acc=0.8306
Epoch 50: loss=0.3643, acc=0.8347
2025-07-03 09:56:48,125 [INFO] Optimizer: Adam, LR: 0.001, Batch size: 32
2025-07-03 09:56:48,126 [INFO] Метрики по эпохам:
2025-07-03 09:56:48,126 [INFO] Accuracy:  0.7541
2025-07-03 09:56:48,129 [INFO] Precision: 0.7597
2025-07-03 09:56:48,132 [INFO] Recall:    0.7457
2025-07-03 09:56:48,136 [INFO] F1-score:  0.7473
2025-07-03 09:56:48,140 [INFO] ROC-AUC:   0.8561
Epoch 10: loss=0.5655, acc=0.7438
Epoch 20: loss=0.4975, acc=0.7893
Epoch 30: loss=0.4568, acc=0.7975
Epoch 40: loss=0.4302, acc=0.7975
Epoch 50: loss=0.4140, acc=0.8017
2025-07-03 09:56:48,755 [INFO] Optimizer: Adam, LR: 0.001, Batch size: 64
2025-07-03 09:56:48,755 [INFO] Метрики по эпохам:
2025-07-03 09:56:48,755 [INFO] Accuracy:  0.8361
2025-07-03 09:56:48,758 [INFO] Precision: 0.8374
2025-07-03 09:56:48,761 [INFO] Recall:    0.8323
2025-07-03 09:56:48,767 [INFO] F1-score:  0.8339
2025-07-03 09:56:48,770 [INFO] ROC-AUC:   0.9199
Epoch 10: loss=0.3866, acc=0.8595
Epoch 20: loss=0.3289, acc=0.8388
Epoch 30: loss=0.3302, acc=0.8471
Epoch 40: loss=0.3416, acc=0.8388
Epoch 50: loss=0.3286, acc=0.8306
2025-07-03 09:56:50,084 [INFO] Optimizer: Adam, LR: 0.01, Batch size: 16
2025-07-03 09:56:50,084 [INFO] Метрики по эпохам:
2025-07-03 09:56:50,084 [INFO] Accuracy:  0.7869
2025-07-03 09:56:50,088 [INFO] Precision: 0.7946
2025-07-03 09:56:50,090 [INFO] Recall:    0.7787
2025-07-03 09:56:50,094 [INFO] F1-score:  0.7810
2025-07-03 09:56:50,098 [INFO] ROC-AUC:   0.8626
Epoch 10: loss=0.3608, acc=0.8388
Epoch 20: loss=0.3444, acc=0.8388
Epoch 30: loss=0.3449, acc=0.8347
Epoch 40: loss=0.3483, acc=0.8430
Epoch 50: loss=0.3394, acc=0.8388
2025-07-03 09:56:50,803 [INFO] Optimizer: Adam, LR: 0.01, Batch size: 32
2025-07-03 09:56:50,803 [INFO] Метрики по эпохам:
2025-07-03 09:56:50,803 [INFO] Accuracy:  0.8033
2025-07-03 09:56:50,807 [INFO] Precision: 0.8164
2025-07-03 09:56:50,811 [INFO] Recall:    0.7938
2025-07-03 09:56:50,814 [INFO] F1-score:  0.7967
2025-07-03 09:56:50,818 [INFO] ROC-AUC:   0.8701
Epoch 10: loss=0.3893, acc=0.8182
Epoch 20: loss=0.3513, acc=0.8471
Epoch 30: loss=0.3407, acc=0.8430
Epoch 40: loss=0.3509, acc=0.8430
Epoch 50: loss=0.3426, acc=0.8471
2025-07-03 09:56:51,405 [INFO] Optimizer: Adam, LR: 0.01, Batch size: 64
2025-07-03 09:56:51,406 [INFO] Метрики по эпохам:
2025-07-03 09:56:51,406 [INFO] Accuracy:  0.8033
2025-07-03 09:56:51,410 [INFO] Precision: 0.8164
2025-07-03 09:56:51,414 [INFO] Recall:    0.7938
2025-07-03 09:56:51,417 [INFO] F1-score:  0.7967
2025-07-03 09:56:51,421 [INFO] ROC-AUC:   0.8734
Epoch 10: loss=0.5254, acc=0.8099
Epoch 20: loss=0.4576, acc=0.8306
Epoch 30: loss=0.4268, acc=0.8182
Epoch 40: loss=0.4998, acc=0.8223
Epoch 50: loss=0.4009, acc=0.8223
2025-07-03 09:56:52,867 [INFO] Optimizer: Adam, LR: 0.1, Batch size: 16
2025-07-03 09:56:52,868 [INFO] Метрики по эпохам:
2025-07-03 09:56:52,868 [INFO] Accuracy:  0.7869
2025-07-03 09:56:52,872 [INFO] Precision: 0.7889
2025-07-03 09:56:52,877 [INFO] Recall:    0.7814
2025-07-03 09:56:52,880 [INFO] F1-score:  0.7832
2025-07-03 09:56:52,884 [INFO] ROC-AUC:   0.8766
Epoch 10: loss=0.3536, acc=0.8306
Epoch 20: loss=0.3702, acc=0.8306
Epoch 30: loss=0.3634, acc=0.8306
Epoch 40: loss=0.3593, acc=0.8430
Epoch 50: loss=0.3695, acc=0.8347
2025-07-03 09:56:53,834 [INFO] Optimizer: Adam, LR: 0.1, Batch size: 32
2025-07-03 09:56:53,834 [INFO] Метрики по эпохам:
2025-07-03 09:56:53,834 [INFO] Accuracy:  0.8033
2025-07-03 09:56:53,838 [INFO] Precision: 0.8164
2025-07-03 09:56:53,840 [INFO] Recall:    0.7938
2025-07-03 09:56:53,845 [INFO] F1-score:  0.7967
2025-07-03 09:56:53,850 [INFO] ROC-AUC:   0.8842
Epoch 10: loss=0.3468, acc=0.8347
Epoch 20: loss=0.3402, acc=0.8430
Epoch 30: loss=0.3416, acc=0.8388
Epoch 40: loss=0.3504, acc=0.8388
Epoch 50: loss=0.3480, acc=0.8388
2025-07-03 09:56:54,398 [INFO] Optimizer: Adam, LR: 0.1, Batch size: 64
2025-07-03 09:56:54,398 [INFO] Метрики по эпохам:
2025-07-03 09:56:54,399 [INFO] Accuracy:  0.8033
2025-07-03 09:56:54,403 [INFO] Precision: 0.8164
2025-07-03 09:56:54,405 [INFO] Recall:    0.7938
2025-07-03 09:56:54,411 [INFO] F1-score:  0.7967
2025-07-03 09:56:54,415 [INFO] ROC-AUC:   0.8658
Epoch 10: loss=0.4154, acc=0.8264
Epoch 20: loss=0.3682, acc=0.8388
Epoch 30: loss=0.3389, acc=0.8512
Epoch 40: loss=0.3492, acc=0.8471
Epoch 50: loss=0.3304, acc=0.8512
2025-07-03 09:56:55,808 [INFO] Optimizer: RMSprop, LR: 0.001, Batch size: 16
2025-07-03 09:56:55,808 [INFO] Метрики по эпохам:
2025-07-03 09:56:55,809 [INFO] Accuracy:  0.7869
2025-07-03 09:56:55,814 [INFO] Precision: 0.8036
2025-07-03 09:56:55,817 [INFO] Recall:    0.7760
2025-07-03 09:56:55,820 [INFO] F1-score:  0.7783
2025-07-03 09:56:55,825 [INFO] ROC-AUC:   0.8777
Epoch 10: loss=0.4093, acc=0.8347
Epoch 20: loss=0.3765, acc=0.8430
Epoch 30: loss=0.3712, acc=0.8595
Epoch 40: loss=0.3595, acc=0.8554
Epoch 50: loss=0.3564, acc=0.8512
2025-07-03 09:56:56,618 [INFO] Optimizer: RMSprop, LR: 0.001, Batch size: 32
2025-07-03 09:56:56,618 [INFO] Метрики по эпохам:
2025-07-03 09:56:56,618 [INFO] Accuracy:  0.7869
2025-07-03 09:56:56,621 [INFO] Precision: 0.8036
2025-07-03 09:56:56,625 [INFO] Recall:    0.7760
2025-07-03 09:56:56,628 [INFO] F1-score:  0.7783
2025-07-03 09:56:56,631 [INFO] ROC-AUC:   0.8831
Epoch 10: loss=0.5492, acc=0.7314
Epoch 20: loss=0.4950, acc=0.7975
Epoch 30: loss=0.4553, acc=0.8017
Epoch 40: loss=0.4281, acc=0.8017
Epoch 50: loss=0.4095, acc=0.8182
2025-07-03 09:56:57,136 [INFO] Optimizer: RMSprop, LR: 0.001, Batch size: 64
2025-07-03 09:56:57,136 [INFO] Метрики по эпохам:
2025-07-03 09:56:57,136 [INFO] Accuracy:  0.8361
2025-07-03 09:56:57,139 [INFO] Precision: 0.8374
2025-07-03 09:56:57,143 [INFO] Recall:    0.8323
2025-07-03 09:56:57,147 [INFO] F1-score:  0.8339
2025-07-03 09:56:57,150 [INFO] ROC-AUC:   0.8874
Epoch 10: loss=0.3465, acc=0.8471
Epoch 20: loss=0.3452, acc=0.8430
Epoch 30: loss=0.3400, acc=0.8306
Epoch 40: loss=0.3338, acc=0.8512
Epoch 50: loss=0.3350, acc=0.8430
2025-07-03 09:56:58,269 [INFO] Optimizer: RMSprop, LR: 0.01, Batch size: 16
2025-07-03 09:56:58,269 [INFO] Метрики по эпохам:
2025-07-03 09:56:58,269 [INFO] Accuracy:  0.8033
2025-07-03 09:56:58,273 [INFO] Precision: 0.8164
2025-07-03 09:56:58,276 [INFO] Recall:    0.7938
2025-07-03 09:56:58,279 [INFO] F1-score:  0.7967
2025-07-03 09:56:58,283 [INFO] ROC-AUC:   0.8788
Epoch 10: loss=0.3374, acc=0.8388
Epoch 20: loss=0.3483, acc=0.8347
Epoch 30: loss=0.3452, acc=0.8430
Epoch 40: loss=0.3490, acc=0.8430
Epoch 50: loss=0.3602, acc=0.8430
2025-07-03 09:56:58,947 [INFO] Optimizer: RMSprop, LR: 0.01, Batch size: 32
2025-07-03 09:56:58,947 [INFO] Метрики по эпохам:
2025-07-03 09:56:58,947 [INFO] Accuracy:  0.8033
2025-07-03 09:56:58,951 [INFO] Precision: 0.8164
2025-07-03 09:56:58,954 [INFO] Recall:    0.7938
2025-07-03 09:56:58,958 [INFO] F1-score:  0.7967
2025-07-03 09:56:58,961 [INFO] ROC-AUC:   0.8669
Epoch 10: loss=0.3387, acc=0.8388
Epoch 20: loss=0.3463, acc=0.8430
Epoch 30: loss=0.3405, acc=0.8388
Epoch 40: loss=0.3436, acc=0.8471
Epoch 50: loss=0.3497, acc=0.8430
2025-07-03 09:56:59,403 [INFO] Optimizer: RMSprop, LR: 0.01, Batch size: 64
2025-07-03 09:56:59,403 [INFO] Метрики по эпохам:
2025-07-03 09:56:59,404 [INFO] Accuracy:  0.8033
2025-07-03 09:56:59,407 [INFO] Precision: 0.8164
2025-07-03 09:56:59,410 [INFO] Recall:    0.7938
2025-07-03 09:56:59,413 [INFO] F1-score:  0.7967
2025-07-03 09:56:59,417 [INFO] ROC-AUC:   0.8712
Epoch 10: loss=0.4933, acc=0.7975
Epoch 20: loss=0.3916, acc=0.8347
Epoch 30: loss=0.3929, acc=0.8140
Epoch 40: loss=0.3986, acc=0.8264
Epoch 50: loss=0.4035, acc=0.8264
2025-07-03 09:57:00,573 [INFO] Optimizer: RMSprop, LR: 0.1, Batch size: 16
2025-07-03 09:57:00,573 [INFO] Метрики по эпохам:
2025-07-03 09:57:00,573 [INFO] Accuracy:  0.7541
2025-07-03 09:57:00,577 [INFO] Precision: 0.7597
2025-07-03 09:57:00,580 [INFO] Recall:    0.7457
2025-07-03 09:57:00,584 [INFO] F1-score:  0.7473
2025-07-03 09:57:00,589 [INFO] ROC-AUC:   0.8160
Epoch 10: loss=0.4461, acc=0.7934
Epoch 20: loss=0.4178, acc=0.8264
Epoch 30: loss=0.4130, acc=0.8347
Epoch 40: loss=0.3828, acc=0.8306
Epoch 50: loss=0.3993, acc=0.8512
2025-07-03 09:57:01,240 [INFO] Optimizer: RMSprop, LR: 0.1, Batch size: 32
2025-07-03 09:57:01,241 [INFO] Метрики по эпохам:
2025-07-03 09:57:01,241 [INFO] Accuracy:  0.7869
2025-07-03 09:57:01,245 [INFO] Precision: 0.7946
2025-07-03 09:57:01,248 [INFO] Recall:    0.7787
2025-07-03 09:57:01,251 [INFO] F1-score:  0.7810
2025-07-03 09:57:01,254 [INFO] ROC-AUC:   0.8831
Epoch 10: loss=0.4028, acc=0.8223
Epoch 20: loss=0.4014, acc=0.8182
Epoch 30: loss=0.3960, acc=0.8223
Epoch 40: loss=0.3680, acc=0.8471
Epoch 50: loss=0.3838, acc=0.8099
2025-07-03 09:57:01,702 [INFO] Optimizer: RMSprop, LR: 0.1, Batch size: 64
2025-07-03 09:57:01,703 [INFO] Метрики по эпохам:
2025-07-03 09:57:01,703 [INFO] Accuracy:  0.8033
2025-07-03 09:57:01,706 [INFO] Precision: 0.8038
2025-07-03 09:57:01,710 [INFO] Recall:    0.7992
2025-07-03 09:57:01,715 [INFO] F1-score:  0.8007
2025-07-03 09:57:01,719 [INFO] ROC-AUC:   0.8528
   optimizer  learning_rate  batch_size  accuracy_val
0        SGD          0.001          16      0.672131
1        SGD          0.001          32      0.721311
2        SGD          0.001          64      0.622951
3        SGD          0.010          16      0.786885
4        SGD          0.010          32      0.786885
5        SGD          0.010          64      0.770492
6        SGD          0.100          16      0.803279
7        SGD          0.100          32      0.803279
8        SGD          0.100          64      0.803279
9       Adam          0.001          16      0.786885
10      Adam          0.001          32      0.754098
11      Adam          0.001          64      0.836066
12      Adam          0.010          16      0.786885
13      Adam          0.010          32      0.803279
14      Adam          0.010          64      0.803279
15      Adam          0.100          16      0.786885
16      Adam          0.100          32      0.803279
17      Adam          0.100          64      0.803279
18   RMSprop          0.001          16      0.786885
19   RMSprop          0.001          32      0.786885
20   RMSprop          0.001          64      0.836066
21   RMSprop          0.010          16      0.803279
22   RMSprop          0.010          32      0.803279
23   RMSprop          0.010          64      0.803279
24   RMSprop          0.100          16      0.754098
25   RMSprop          0.100          32      0.786885
26   RMSprop          0.100          64      0.803279
Базовая модель:
Epoch 10: loss=0.5267, acc=0.7521
Epoch 20: loss=0.4162, acc=0.8182
Epoch 30: loss=0.3789, acc=0.8388
Epoch 40: loss=0.3659, acc=0.8471
Epoch 50: loss=0.3518, acc=0.8471
Метрики на валидации:
2025-07-03 09:59:24,236 [INFO] Метрики по эпохам:
2025-07-03 09:59:24,237 [INFO] Accuracy:  0.7869
2025-07-03 09:59:24,242 [INFO] Precision: 0.8036
2025-07-03 09:59:24,247 [INFO] Recall:    0.7760
2025-07-03 09:59:24,250 [INFO] F1-score:  0.7783
2025-07-03 09:59:24,253 [INFO] ROC-AUC:   0.8766

Модель с расширенными признаками:
Epoch 10: loss=0.3474, acc=0.8471
Epoch 20: loss=0.3360, acc=0.8430
Epoch 30: loss=0.3182, acc=0.8512
Epoch 40: loss=0.3164, acc=0.8554
Epoch 50: loss=0.3190, acc=0.8595
Метрики на валидации:
2025-07-03 09:59:24,862 [INFO] Метрики по эпохам:
2025-07-03 09:59:24,862 [INFO] Accuracy:  0.8033
2025-07-03 09:59:24,865 [INFO] Precision: 0.8164
2025-07-03 09:59:24,867 [INFO] Recall:    0.7938
2025-07-03 09:59:24,871 [INFO] F1-score:  0.7967
2025-07-03 09:59:24,875 [INFO] ROC-AUC:   0.8755
