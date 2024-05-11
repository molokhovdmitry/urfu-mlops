 # lab1 - Простейший пайплайн

<details>

## data_creation.py
Скрипт для создания данных о дневной температуре за 7 лет. Создает три набора данных с разным уровнем шума. 20% каждого набора данных сохраняются как тестовые.

## model_preprocessing.py
В данном скрипте определен объект `preprocessors` класса `ColumnTransformer` для предобработки данных.

## model_preparation.py
Скрипт для предобработки данных, тренировки и сохранения модели в `model.pt`.

## model_testing.py
Скрипт для предсказания температуры на тестовых данных и вывода `MSE`.

## pipeline.sh
Скрипт для запуска пайплайна.

## notebook.ipynb
Демонстрационный ноутбук.

</details>

# lab2 - Jenkins

<details>

## data.py
Скрипт для загрузки датасета [Real vs Fake Faces - 10k](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k/data) с `Kaggle` и функция для создания `DataLoader` для тренировочной, валидационной и тестовой выборок.

## train_model.py
Скрипт для определения архитектуры и тренировки простейшей сверточной нейросети. При тренировке выводит потерю на тренировочных и валидационных данных.

## test_model.py
Скрипт для получения предсказаний на тестовых данных и вывода метрики `accuracy`. Также скрипт выводит `accuracy` для случайных предсказаний.

## Dockerfile
Создает образ `Jenkins` и устанавливает зависимости.

## Jenkinsfile
Запускает скрипты.

## Оценка качества модели
Данной архитектуры достаточно для переобучения модели на тренировочных данных, но потери на валидационных данных при тренировке не снижались. Качество предсказаний модели на тестовых данных не выше, чем при случайных предсказаниях. Для решения задачи требуется модель более сложной архитектуры.

## Запуск контейнера
```
docker run -p 8080:8080 -p 50000:50000 -v /home/$USER/.kaggle/kaggle.json:/var/jenkins_home/.kaggle/kaggle.json lab2
```

</details>

# lab3 - Docker

<details>

Реализованы контейнер `lab3-model`, тренирующий модель `ResNet50` и контейнер `lab3-app` - приложение на `Streamlit`, использующее эту модель. 

## `/model/`

Содержит файлы образа [molokhovdmitry/lab3-model](https://hub.docker.com/r/molokhovdmitry/lab3-model/tags) для тренировки `ResNet50` на датасете по распознаванию мусора.

### data.py
Скрипт для загрузки c `Kaggle` датасета [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) и функция для создания DataLoader для тренировочной, валидационной и тестовой выборок.
### train_model.py
Скрипт для тренировки `ResNet50` на датасете и сохранения модели в общую с контейнером `lab3-app` папку `models` при улучшении метрики `Weighted F1 score` на валидационной выборке.
### test_model.py
Скрипт для получения предсказаний на тестовых данных, вывода метрики `Weighted F1 score` и `Classification report`.
### pipeline.sh
Скрипт для запуска пайплайна в контейнере.

## `/app/`
Содержит файлы образа [molokhovdmitry/lab3-app](https://hub.docker.com/r/molokhovdmitry/lab3-app/tags), который развертывает веб-приложение `Streamlit` для распознавания типа мусора по картинке, использующий модель, натренированную контейнером `lab3-model`.

### app.py
Файл приложения `Streamlit`.

## docker-compose.yml
Файл для запуска контейнеров.

## .github/workflows/docker-images.yml
Workflow для запуска и загрузки образов в [dockerhub](https://hub.docker.com/u/molokhovdmitry) с привязкой имени тэга к версии сборки.

</details>

# lab4 - DVC

<details>

Скрипты для обработки датасета [US Accidents (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

### cluster_coords.py
Скрипт для `KMeans` кластеризации по признакам `Start_Lat` и `Start_Lng`.
### remove_nans.py
Скрипт для заполнения пропущенных значений средним для числовых признаков, самым частым значением для категориальных.
### ohe.py
Скрипт для `One-hot` кодирования признаков `Source` и `State`.

</details>

# lab5 - PyTest

<details>

## lab.ipynb
- Скачивает данные с `Kaggle` соревнования [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)
- Делит данные на 4 датасета, в одном из датасетов к признакам `pickup_longitude` `pickup_latitude` `dropoff_longitude` `dropoff_latitude` добавляется шум. Сохраняет датасеты.
- Предобрабатывает данные и тренирует модель линейной регрессии на первом наборе данных. Сохраняет пайплайн модели с помощью `joblib`.
- Считает метрику `RMSLE` для предсказаний на остальных наборах данных.
- Запускает `PyTest`, который обнаруживает большую метрику `RMSLE` на наборе данных с шумом.

## test_model.py
Тесты для сравнения метрики RMSLE на наборах данных с заданным `THRESHOLD_RMSLE`.

</details>
