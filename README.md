# Запуск STG-NF на собственных данных и визуализация

## Шаг 1: Генерация данных

1. Запускаем скрипт `generate_data.py` на своем видео с толпой людей.

2. После запуска скрипт создаст файл `08_0044_alphapose_tracked_person.json` (название временное, но важно, чтобы STG-NF правильно его подхватил).

![image](https://github.com/user-attachments/assets/3aacbb76-fc9a-4e7c-940b-5bd8252ee690)


## Шаг 2: Запускаем модель

1. Теперь запускаем команду:

   ```bash
   python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar

В итоге у нас сгенерируется файл `normality_scores.json`

## Шаг 3: Визуализация

Далее запускаем `visualization.py` (в нем должны быть прописан корректные пути до файла `normality_scores.json` и до файла видео в формате `.mp4`)

И получаем визуализацию наших аномалий на видео!

## Запуск STG-NF на собственных данных и визуализация в один шаг

   ```bash
   python generate_data.py && python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar && python visualization.py